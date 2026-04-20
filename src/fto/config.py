from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Callable
from fto.adapters.node.node import NodeAdapter
from fto.faults import Fault
from fto.recovery import Restart
from fto.recovery.checkpoint import Checkpoint


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
    fault: Fault | None = None
    restart: Restart | None = None
    get_edge_propagator: Callable | None = None
    suppress_edge_propagator: Callable | None = None
    restore_edge_propagator: Callable | None = None
    logger: Any = None
    checkpoint: Checkpoint | None = None
