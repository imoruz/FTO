from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Callable
from fto.adapters.edge import EdgeSuppressor
from fto.faults import Fault
from fto.detection.observer import Observer
from fto.recovery import Restart
from fto.recovery.checkpoint import Checkpoint
from fto.adapters.probes import ObservationProbe


class RestartMode(StrEnum):
    NO_HISTORY = auto()
    ALL_HISTORY = auto()
    REFINED_HISTORY = auto()


@dataclass
class FTOConfig:
    fault: Fault | None = None
    restart: Restart | None = None
    edge_suppressor: EdgeSuppressor = field(default_factory=EdgeSuppressor)
    logger: Any = None
    checkpoint: Checkpoint | None = None
    instrumentation: Callable[..., Any] | None = None
    observer: Observer | None = None
    probe: ObservationProbe | None = None
