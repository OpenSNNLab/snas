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
