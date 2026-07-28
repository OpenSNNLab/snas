from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Graph:
    """
       [ Input A ] ───┐
                      ▼
                   [ Node C ] ───► [ Output D ]
                      ▲
       [ Input B ] ───┘


    routing_map:

          DESTINATION KEY              SOURCE DEPENDENCIES
        ┌─────────────────┐           ┌────────────────────┐
        │    Output D     │ ◄─────────┤     [ Node C ]     │
        ├─────────────────┤           ├────────────────────┤
        │    Node C       │ ◄─────────┤[ Input A, Input B ]│
        ├─────────────────┤           ├────────────────────┤
        │    Input A      │ ◄─────────┤     [ ]            │
        ├─────────────────┤           ├────────────────────┤
        │    Input B      │ ◄─────────┤     [ ]            │
        └─────────────────┘           └────────────────────┘
    """

    routing_map: Dict[str, List[str]] = field(default_factory=dict)

    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)

    def _get_insertion_index(
        self,
        dependencies: List[str],
        dst: str,
        before: Optional[str] = None,
        after: Optional[str] = None,
    ) -> int:
        if before is not None and after is not None:
            if before not in dependencies:
                raise ValueError(
                    f"Target node '{before}' is not a dependency of '{dst}'."
                )
            if after not in dependencies:
                raise ValueError(
                    f"Target node '{after}' is not a dependency of '{dst}'."
                )

            idx_before = dependencies.index(before)
            idx_after = dependencies.index(after)

            if idx_after >= idx_before:
                raise ValueError(
                    f"Contradictory constraints: '{after}' appears at or after '{before}' "
                    f"in the dependencies of '{dst}' ({dependencies})."
                )

            return idx_before

        elif before is not None:
            if before not in dependencies:
                raise ValueError(
                    f"Target node '{before}' is not a dependency of '{dst}'."
                )
            return dependencies.index(before)

        elif after is not None:
            if after not in dependencies:
                raise ValueError(
                    f"Target node '{after}' is not a dependency of '{dst}'."
                )
            return dependencies.index(after) + 1

        return len(dependencies)  # Default: append at the end

    def add_edge(
        self,
        src: str,
        dst: str,
        before: Optional[str] = None,
        after: Optional[str] = None,
    ) -> None:
        if src not in self.routing_map:
            self.routing_map[src] = []
        if dst not in self.routing_map:
            self.routing_map[dst] = []

        dependencies = self.routing_map[dst]

        if src in dependencies:  # Prevent duplicate edges
            return

        idx = self._get_insertion_index(dependencies, dst, before, after)
        dependencies.insert(idx, src)

    def remove_node(self, node: str) -> None:
        if node in self.routing_map:
            del self.routing_map[node]

        for dependencies in self.routing_map.values():
            if node in dependencies:
                dependencies.remove(node)

    def compute_execution_order(self) -> None:
        in_degree: Dict[str, int] = {
            node: len(deps) for node, deps in self.routing_map.items()
        }

        dependents: Dict[str, List[str]] = {node: [] for node in self.routing_map}

        for dst, deps in self.routing_map.items():
            for src in deps:
                if src not in dependents:
                    dependents[src] = []
                    in_degree[src] = 0
                dependents[src].append(dst)

        queue: deque[str] = deque(
            [node for node, degree in in_degree.items() if degree == 0]
        )
        order: List[str] = []

        while queue:
            current = queue.popleft()
            order.append(current)

            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(in_degree):
            unresolved = {node for node, degree in in_degree.items() if degree > 0}
            raise RuntimeError(
                "Cyclic dependency detected."
                f" The following nodes could not be resolved: {unresolved}"
            )

        self.execution_order = order
