import json
from typing import Any, Dict, Iterator

import torch.nn as nn


class BasePlugin(nn.Module):
    default_attach_mode: str = "after"

    def __init__(self) -> None:
        super().__init__()
        self.execution_count: int = 0

    @property
    def result(self) -> str:
        if self.execution_count == 0:
            return "No data"
        return f"{self.execution_count}x runs"

    def clear(self) -> None:
        self.execution_count = 0


class BaseWrapper(BasePlugin):
    default_attach_mode: str = "wrap"

    def __init__(self, target_module: nn.Module) -> None:
        super().__init__()
        self.target_module = target_module

    def unwrap(self) -> Iterator[nn.Module]:
        current = self
        while isinstance(current, BaseWrapper):
            yield current
            current = current.target_module

        if current is not None:
            yield current

    @property
    def root_module(self) -> nn.Module:
        *_, root = self.unwrap()
        return root
