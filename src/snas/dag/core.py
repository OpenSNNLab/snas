import copy
import re
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch.fx as fx
import torch.nn as nn

from ..plugins.base import BasePlugin
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
        self.args_schema = args_schema if args_schema is not None else tuple()
        self.kwargs_schema = kwargs_schema if kwargs_schema is not None else dict()
        self.is_method = is_method

    @staticmethod
    def _resolve_schema(schema: Any, input_iter: Iterator[Any]) -> Any:
        if isinstance(schema, NodePlaceholder):
            return next(input_iter)
        if isinstance(schema, tuple):
            return tuple(OpModule._resolve_schema(x, input_iter) for x in schema)
        if isinstance(schema, list):
            return [OpModule._resolve_schema(x, input_iter) for x in schema]
        if isinstance(schema, dict):
            return {
                k: OpModule._resolve_schema(v, input_iter) for k, v in schema.items()
            }
        return schema

    def forward(self, *inputs: Any) -> Any:
        input_iter = iter(inputs)

        resolved_args = self._resolve_schema(self.args_schema, input_iter)
        resolved_kwargs = self._resolve_schema(self.kwargs_schema, input_iter)

        if self.is_method:
            obj = resolved_args[0]
            method = getattr(obj, self.target)
            return method(*resolved_args[1:], **resolved_kwargs)

        return self.target(*resolved_args, **resolved_kwargs)


class DAG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.topology = Graph()
        self.module_pool = nn.ModuleDict()
        self._is_locked = False

    def forward(self, *args, **kwargs) -> Any:
        total_inputs = len(args) + len(kwargs)
        expected_inputs = len(self.topology.input_keys)
        if total_inputs > expected_inputs:
            raise ValueError(f"Expected {expected_inputs} inputs, got {total_inputs}")

        cache: Dict[str, Any] = {}

        for i, key in enumerate(self.topology.input_keys):
            if i < len(args):
                cache[key] = args[i]
            elif key in kwargs:
                cache[key] = kwargs[key]
            else:
                raise ValueError(f"Missing required input for graph node: '{key}'")

        for node_name in self.topology.execution_order:
            if node_name in cache:
                continue

            module = self.module_pool[node_name]

            deps = self.topology.routing_map.get(node_name, [])
            inputs = [cache[dep] for dep in deps]

            cache[node_name] = module(*inputs)

        outputs = tuple(cache[key] for key in self.topology.output_keys)

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    @contextmanager
    def clone(self):
        new_dag = DAG()
        new_dag.topology = copy.deepcopy(self.topology)
        new_dag.module_pool = copy.deepcopy(self.module_pool)
        new_dag._is_locked = False

        try:
            yield new_dag
        finally:
            new_dag.topology.compute_execution_order()
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
                    if target in deps:
                        deps[deps.index(target)] = new_node

                self.topology.routing_map[new_node] = [target]

                if target in self.topology.output_keys:
                    idx = self.topology.output_keys.index(target)
                    self.topology.output_keys[idx] = new_node

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

            if not node_name.endswith(suffix):
                continue

            target_node = node_name[: -len(suffix)]

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

    def draw(
        self,
        backend: str = "graphviz",
        filepath: Optional[str] = None,
        format: str = "png",
    ) -> Any:
        if backend == "ascii":
            return self._draw_ascii()
        elif backend == "graphviz":
            return self._draw_graphviz(filepath, format)
        else:
            raise ValueError(
                f"Unsupported backend '{backend}'. Use 'graphviz' or 'ascii'."
            )

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
        dag._is_locked = True

        return dag

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
            lines.append(f"{prefix}{connector}{node}")

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
            if not is_input:
                mod = (
                    self.module_pool[node_key] if node_key in self.module_pool else None
                )

                if isinstance(mod, OpModule):
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

            return f"<{node_key}{mod_label}{tag}>"

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
