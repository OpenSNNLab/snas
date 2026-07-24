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

            # Impossible (after="X" but before="Y" fo [Y, X])
            if idx_after >= idx_before:
                raise ValueError(
                    f"Contradictory constraints: '{after}' appears at or after '{before}' "
                    f"in the dependencies of '{dst}' ({dependencies})."
                )

            dependencies.insert(idx_before, src)

        elif before is not None:
            if before not in dependencies:
                raise ValueError(
                    f"Target node '{before}' is not a dependency of '{dst}'."
                )
            idx = dependencies.index(before)
            dependencies.insert(idx, src)

        elif after is not None:
            if after not in dependencies:
                raise ValueError(
                    f"Target node '{after}' is not a dependency of '{dst}'."
                )
            idx = dependencies.index(after)
            dependencies.insert(idx + 1, src)

        else:  # Default: append at the end
            dependencies.append(src)

    def remove_node(self, node: str) -> None:
        if node in self.routing_map:
            del self.routing_map[node]

        for dest, dependencies in self.routing_map.items():
            self.routing_map[dest] = [dep for dep in dependencies if dep != node]
