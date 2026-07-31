import json
from abc import ABC
from dataclasses import asdict, dataclass

import torch.nn as nn


@dataclass
class BaseResult(ABC):
    execution_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def to_console(self) -> str:
        return f"Execs: {self.execution_count}"


class BasePlugin(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._result = None

    @property
    def result(self) -> BaseResult:
        if getattr(self, "_result", None) is None:
            raise RuntimeError(f"Data not yet captured by {self.__class__.__name__}.")
        return self._result

    def clear(self) -> None:
        self._result = None
