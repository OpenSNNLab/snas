import copy
import os
import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree

from ..plugins.base import BasePlugin, BaseWrapper
from .structure import Graph
from .tracing import SymbolicTracer


class NodePlaceholder:
    def __repr__(self) -> str:
        return "<NodePlaceholder>"


class OpModule(nn.Module):
    def __init__(
        self,
        target: Any,
        args_schema: Any = None,
        kwargs_schema: Any = None,
        is_method: bool = False,
    ) -> None:
        super().__init__()
        self.target = target
        self.is_method = is_method

        self.args_flat, self.args_spec = pytree.tree_flatten(args_schema or tuple())
        self.kwargs_flat, self.kwargs_spec = pytree.tree_flatten(
            kwargs_schema or dict()
        )

        self._arg_placeholders = [
            i for i, x in enumerate(self.args_flat) if isinstance(x, NodePlaceholder)
        ]
        self._kwarg_placeholders = [
            i for i, x in enumerate(self.kwargs_flat) if isinstance(x, NodePlaceholder)
        ]

    def forward(self, *inputs: Any) -> Any:
        input_iter = iter(inputs)

        resolved_args_flat = list(self.args_flat)
        for idx in self._arg_placeholders:
            resolved_args_flat[idx] = next(input_iter)

        resolved_kwargs_flat = list(self.kwargs_flat)
        for idx in self._kwarg_placeholders:
            resolved_kwargs_flat[idx] = next(input_iter)

        resolved_args = pytree.tree_unflatten(resolved_args_flat, self.args_spec)
        resolved_kwargs = pytree.tree_unflatten(resolved_kwargs_flat, self.kwargs_spec)

        if self.is_method:
            obj, *rest_args = resolved_args
            return getattr(obj, self.target)(*rest_args, **resolved_kwargs)

        return self.target(*resolved_args, **resolved_kwargs)


class DAG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.topology = Graph()
        self.module_pool = nn.ModuleDict()
        self._is_locked = False
        self._execution_plan = []

    def forward(self, *args, **kwargs) -> Any:
        expected_inputs = len(self.topology.input_keys)
        if len(args) + len(kwargs) > expected_inputs:
            raise ValueError(
                f"Expected {expected_inputs} inputs, got {len(args) + len(kwargs)}"
            )

        cache: Dict[str, Any] = dict(zip(self.topology.input_keys, args))
        for key in self.topology.input_keys[len(args) :]:
            try:
                cache[key] = kwargs[key]
            except KeyError:
                raise ValueError(f"Missing required input for graph node: '{key}'")

        for node_name, module, deps in self._execution_plan:
            inputs = tuple(cache[dep] for dep in deps)
            cache[node_name] = module(*inputs)

        outputs = tuple(cache[key] for key in self.topology.output_keys)
        return outputs[0] if len(outputs) == 1 else outputs

    @contextmanager
    def clone(self):
        new_dag = DAG()
        new_dag.topology = copy.deepcopy(self.topology)
        new_dag.module_pool = copy.deepcopy(self.module_pool)
        new_dag._is_locked = False

        new_dag._compile_execution_plan()

        try:
            yield new_dag
        finally:
            new_dag.topology.compute_execution_order()
            new_dag._compile_execution_plan()
            new_dag._is_locked = True

    def insert(self, module: nn.Module, after: Union[str, List[str]]) -> "DAG":
        if self._is_locked:
            raise RuntimeError(
                "This DAG is locked! "
                "Modifications must be done inside the `with dag.clone():` context."
            )

        targets = self._resolve_target_names(after)

        with self.topology:
            for target in targets:
                module_name = module.__class__.__name__
                new_node = f"{target}_{module_name}"

                self.module_pool[new_node] = copy.deepcopy(module)

                for _, deps in self.topology.routing_map.items():
                    for i, dep in enumerate(deps):
                        if dep == target:
                            deps[i] = new_node

                self.topology.routing_map[new_node] = [target]

                for i, out_key in enumerate(self.topology.output_keys):
                    if out_key == target:
                        self.topology.output_keys[i] = new_node

        self._compile_execution_plan()
        return self

    def get_plugins(
        self,
        plugin_type: type = BasePlugin,
        group_by: str = "type",
    ) -> Dict[str, Any]:
        if not issubclass(plugin_type, BasePlugin):
            raise TypeError(f"{plugin_type.__name__} must inherit from BasePlugin.")

        collected_plugins = {}

        for node_name, plugin_instance in self.module_pool.items():
            if not isinstance(plugin_instance, plugin_type):
                continue

            plugin_class_name = plugin_instance.__class__.__name__
            suffix = f"_{plugin_class_name}"

            if node_name.endswith(suffix):
                target_node = node_name[: -len(suffix)]
            else:
                target_node = node_name

            if plugin_type is not BasePlugin:
                collected_plugins[target_node] = plugin_instance
                continue

            if group_by == "type":
                if plugin_class_name not in collected_plugins:
                    collected_plugins[plugin_class_name] = {}
                collected_plugins[plugin_class_name][target_node] = plugin_instance

            elif group_by == "layer":
                if target_node not in collected_plugins:
                    collected_plugins[target_node] = {}
                collected_plugins[target_node][plugin_class_name] = plugin_instance

            else:
                raise ValueError("group_by must be 'type' or 'layer'")

        return collected_plugins

    def draw(self, filepath: Optional[str] = None) -> Any:
        if filepath is None:
            ascii_tree = self._draw_ascii()
            print(ascii_tree)
            return ascii_tree

        base_name, ext = os.path.splitext(filepath)
        fmt = ext.lstrip(".").lower()

        if not fmt:
            fmt = "png"

        if fmt not in ["svg", "png", "pdf", "jpg", "jpeg"]:
            raise ValueError(f"Unsupported file extension '.{fmt}'. Use .svg or .png")

        return self._draw_graphviz(filepath=base_name, format=fmt)

    def to_module(self) -> fx.GraphModule:
        graph = fx.Graph()
        cache: Dict[str, fx.Node] = {}

        for key in self.topology.input_keys:
            cache[key] = graph.placeholder(key)

        for node in self.topology.execution_order:
            if node in self.topology.input_keys:
                continue

            mod = self.module_pool[node]
            inputs = [
                cache[in_key] for in_key in self.topology.routing_map.get(node, [])
            ]

            if isinstance(mod, OpModule):
                input_iter = iter(inputs)

                resolved_args = OpModule._resolve_schema(mod.args_schema, input_iter)
                resolved_kwargs = OpModule._resolve_schema(
                    mod.kwargs_schema, input_iter
                )

                if mod.is_method:
                    cache[node] = graph.call_method(
                        mod.target, tuple(resolved_args), resolved_kwargs
                    )
                else:
                    cache[node] = graph.call_function(
                        mod.target, tuple(resolved_args), resolved_kwargs
                    )
            else:
                cache[node] = graph.call_module(f"module_pool.{node}", tuple(inputs))

        out_nodes = tuple(cache[k] for k in self.topology.output_keys)
        graph.output(out_nodes[0] if len(out_nodes) == 1 else out_nodes)

        return fx.GraphModule(self, graph)

    @classmethod
    def from_module(cls, module: nn.Module) -> "DAG":
        if not isinstance(module, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(module)}")

        graph = SymbolicTracer().trace(module)
        module_dict = dict(module.named_modules())
        dag = cls()

        with dag.topology:
            for node in graph.nodes:
                if node.op == "placeholder":
                    dag.topology.input_keys.append(node.name)
                    if node.name not in dag.topology.routing_map:
                        dag.topology.routing_map[node.name] = []

                elif node.op == "call_module":
                    target_mod = module_dict.get(str(node.target))
                    if target_mod is None:
                        raise RuntimeError(
                            f"Module {node.target} missing from ModuleDict."
                        )

                    dag.module_pool[node.name] = target_mod
                    _, _, deps = cls._parse_args(node.args, node.kwargs)

                    for dep in deps:
                        dag.topology.add_edge(src=dep, dst=node.name)

                    if node.name not in dag.topology.routing_map:
                        dag.topology.routing_map[node.name] = []

                elif node.op == "call_function":
                    args_schema, kwargs_schema, deps = cls._parse_args(
                        node.args, node.kwargs
                    )
                    dag.module_pool[node.name] = OpModule(
                        node.target, args_schema, kwargs_schema
                    )

                    for dep in deps:
                        dag.topology.add_edge(src=dep, dst=node.name)

                    if node.name not in dag.topology.routing_map:
                        dag.topology.routing_map[node.name] = []

                elif node.op == "call_method":
                    args_schema, kwargs_schema, deps = cls._parse_args(
                        node.args, node.kwargs
                    )
                    dag.module_pool[node.name] = OpModule(
                        str(node.target), args_schema, kwargs_schema, is_method=True
                    )

                    for dep in deps:
                        dag.topology.add_edge(src=dep, dst=node.name)

                    if node.name not in dag.topology.routing_map:
                        dag.topology.routing_map[node.name] = []

                elif node.op == "output":
                    output_args = node.args[0]
                    if isinstance(output_args, tuple):
                        dag.topology.output_keys = [
                            n.name for n in output_args if isinstance(n, fx.Node)
                        ]
                    elif isinstance(output_args, fx.Node):
                        dag.topology.output_keys = [output_args.name]
                    else:
                        raise TypeError(f"Unsupported output type: {type(output_args)}")

        dag._compile_execution_plan()
        dag._is_locked = True

        return dag

    def _compile_execution_plan(self):
        self._execution_plan = []
        for node_name in self.topology.execution_order:
            if node_name in self.topology.input_keys:
                continue

            module = self.module_pool[node_name]
            deps = tuple(self.topology.routing_map.get(node_name, []))
            self._execution_plan.append((node_name, module, deps))

    @staticmethod
    def _parse_args(args: Any, kwargs: Any) -> Tuple[Any, Any, List[str]]:
        dependencies: List[str] = []

        def map_fn(n: Any) -> Any:
            if isinstance(n, fx.Node):
                dependencies.append(n.name)
                return NodePlaceholder()
            if isinstance(n, tuple):
                return tuple(map_fn(x) for x in n)
            if isinstance(n, list):
                return [map_fn(x) for x in n]
            if isinstance(n, dict):
                return {k: map_fn(v) for k, v in n.items()}
            return n

        args_schema = map_fn(args)
        kwargs_schema = map_fn(kwargs)
        return args_schema, kwargs_schema, dependencies

    def _resolve_target_names(self, pattern: Union[str, List[str]]) -> List[str]:
        if isinstance(pattern, list):
            return pattern

        if "*" in pattern:
            regex = re.compile(pattern.replace("*", ".*"))
            return [n for n in self.topology.routing_map.keys() if regex.match(n)]

        if pattern not in self.topology.routing_map:
            raise ValueError(f"Target node '{pattern}' not found in graph.")

        return [pattern]

    def _draw_ascii(self) -> str:
        lines: List[str] = []
        visited = set()

        def dfs(node: str, prefix: str, is_last: bool) -> None:
            connector = "└── " if is_last else "├── "

            stats_label = ""
            mod = self.module_pool[node] if node in self.module_pool else None

            if isinstance(mod, BaseWrapper):
                wrapper_name = mod.__class__.__name__
                if hasattr(mod, "_result") and mod._result is not None:
                    stats = mod.result.to_console()
                    stats_label = f"  [{wrapper_name} -> {stats}]"
                else:
                    stats_label = f"  [Wrapped: {wrapper_name}]"

            lines.append(f"{prefix}{connector}{node}{stats_label}")

            if node in visited:
                lines[-1] += " (shared constraint)"
                return
            visited.add(node)

            deps = self.topology.routing_map.get(node, [])
            for i, dep in enumerate(deps):
                extension = "    " if is_last else "│   "
                dfs(dep, prefix + extension, i == (len(deps) - 1))

        lines.append("[ DAG Output Dependencies ]")
        for i, out_node in enumerate(self.topology.output_keys):
            dfs(out_node, "", i == (len(self.topology.output_keys) - 1))

        return "\n".join(lines)

    def _draw_graphviz(self, filepath: Optional[str], format: str) -> Any:
        try:
            import graphviz
        except ImportError:
            raise ImportError(
                "The 'graphviz' library is required for this backend. "
                "Install it using: pip install graphviz"
            )

        dot = graphviz.Digraph(comment="DAG Topology")

        dot.attr(
            rankdir="LR",  # Left to Right pipeline
            splines="spline",  # Smooth edges
            nodesep="0.6",  # Vertical spacing
            ranksep="0.8",  # Horizontal spacing
            fontname="Helvetica",
        )

        dot.attr("edge", color="#94a3b8", penwidth="1.5", arrowsize="0.8")

        def get_node_label(node_key: str, is_input: bool, is_output: bool) -> str:
            mod_label = ""
            stats_label = ""
            if not is_input:
                mod = (
                    self.module_pool[node_key] if node_key in self.module_pool else None
                )

                if isinstance(mod, BaseWrapper):
                    tgt_name = mod.target_module.__class__.__name__
                    wrapper_name = mod.__class__.__name__
                    mod_label = f"<BR/><FONT POINT-SIZE='10' COLOR='#64748b'>{tgt_name} (Wrapped: {wrapper_name})</FONT>"

                    if hasattr(mod, "_result") and mod._result is not None:
                        stats = mod.result.to_console()
                        stats_label = f"<BR/><FONT POINT-SIZE='9' COLOR='#b91c1c'>[{stats}]</FONT>"

                elif isinstance(mod, OpModule):
                    tgt_name = getattr(mod.target, "__name__", str(mod.target))
                    mod_label = (
                        f"<BR/><FONT POINT-SIZE='10' COLOR='#64748b'>{tgt_name}</FONT>"
                    )
                elif mod is not None:
                    mod_label = f"<BR/><FONT POINT-SIZE='10' COLOR='#64748b'>{mod.__class__.__name__}</FONT>"

            tag = ""
            if is_input:
                tag = "<BR/><FONT POINT-SIZE='9' COLOR='#166534'>[Input]</FONT>"
            elif is_output:
                tag = "<BR/><FONT POINT-SIZE='9' COLOR='#1e3a8a'>[Output]</FONT>"

            return f"<{node_key}{mod_label}{tag}{stats_label}>"

        for key in self.topology.execution_order:
            is_input = key in self.topology.input_keys
            is_output = key in self.topology.output_keys

            label = get_node_label(key, is_input, is_output)

            if is_input:
                dot.node(
                    key,
                    label=label,
                    shape="rect",
                    style="rounded,filled",
                    fillcolor="#dcfce7",
                    color="#22c55e",
                    penwidth="2",
                    fontname="Helvetica",
                )
            elif is_output:
                dot.node(
                    key,
                    label=label,
                    shape="rect",
                    style="rounded,filled",
                    fillcolor="#dbeafe",
                    color="#3b82f6",
                    penwidth="3",
                    fontname="Helvetica",
                )
            else:
                dot.node(
                    key,
                    label=label,
                    shape="rect",
                    style="rounded,filled",
                    fillcolor="#f8fafc",
                    color="#cbd5e1",
                    penwidth="1.5",
                    fontname="Helvetica",
                )

        for dst, deps in self.topology.routing_map.items():
            for src in deps:
                dot.edge(src, dst)

        if filepath:
            dot.render(filepath, format=format, cleanup=True)

        return dot
