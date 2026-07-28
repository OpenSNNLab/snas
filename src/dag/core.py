from typing import Any, Dict, Iterator, List, Tuple

import torch.fx as fx
import torch.nn as nn

from structure import Graph
from tracing import SymbolicTracer


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

        return dag
