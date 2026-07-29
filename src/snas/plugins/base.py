from typing import Any

import torch.nn as nn


class BasePlugin(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._result = None

    @property
    def result(self) -> Any:
        if not hasattr(self, "_result") or self._result is None:
            raise RuntimeError(f"Result not yet captured by {self.__class__.__name__}.")
        return self._result

    def clear(self) -> None:
        self._result = None
