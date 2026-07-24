from typing import Any, Dict, Optional, Tuple

import torch
import torch.fx as fx
import torch.nn as nn


class ControlFlowDetected(Exception):
    def __init__(self, module_type: type) -> None:
        self.module_type = module_type


class SymbolicTracer(fx.Tracer):
    """
               ┌──────────────────────┐
               │     trace(module)    │
               └───────────┬──────────┘
                           │
                           ▼
               ┌──────────────────────┐
    ┌─────────►│  _active_modules     │
    │          │  .clear()            │
    │          └───────────┬──────────┘
    │                      │
    │                      ▼
    │          ┌──────────────────────┐
    │          │ Execute              │
    │          │ super().trace(...)   │
    │          └───────────┬──────────┘
    │                      │
    │            Successful?
    │            ├─── [ YES ] ───►  [ Return fx.Graph ]
    │            │
    │            [ NO ] (Throws torch.fx.proxy.TraceError)
    │            │
    │            ▼
    │          ┌──────────────────────┐
    │          │ Intercepted via      │
    │          │ call_module catch    │
    │          └───────────┬──────────┘
    │                      │
    │                      ▼
    │          ┌──────────────────────┐
    │          │ Raise custom         │
    │          │ ControlFlowDetected  │
    │          └───────────┬──────────┘
    │                      │
    │                      ▼
    │          ┌──────────────────────┐
    │          │ Append failed module │
    │          │ to blacklist_leaves  │
    └──────────┴──────────────────────┘
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blacklist_leaves = set()

    def trace(
        self,
        root: nn.Module,
        concrete_args: Optional[Dict[str, Any]] = None,
    ) -> fx.Graph:
        while True:
            try:
                return super().trace(root, concrete_args=concrete_args)
            except ControlFlowDetected as e:
                self.blacklist_leaves.add(e.module_type)

    def is_leaf_module(self, m: nn.Module, module_qualname: str) -> bool:
        if type(m) in self.blacklist_leaves:
            return True

        if not hasattr(m, "__module__"):
            return True

        if super().is_leaf_module(m, module_qualname):
            return True

        return False

    def call_module(
        self,
        m: nn.Module,
        forward: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        try:
            return super().call_module(m, forward, args, kwargs)
        except torch.fx.proxy.TraceError:
            raise ControlFlowDetected(type(m))
