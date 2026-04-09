from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Callable
from fto.adapters.node.node import NodeAdapter
from fto.const import DEFAULT_INJECTED_PROMPT
from fto.faults import Fault


class RestartMode(StrEnum):
    NO_HISTORY = auto()
    ALL_HISTORY = auto()
    REFINED_HISTORY = auto()


class CircuitState(StrEnum):
    OPEN = auto()
    CLOSED = auto()


@dataclass
class FTOConfig:

    node_adapter: NodeAdapter
    patch_target: Any
    patch_method: Any
    fault: Fault = None
    restart_mode: RestartMode = None
    get_edge_propagator: Callable[[Any], Any] = None
    suppress_edge_propagator: Callable[[Any], None] = None
    restore_edge_propagator: Callable[[Any, Any], None] = None
    logger: Any = None

    def resolve_patch_target(self):
        if self.patch_target:
            return self.patch_target
