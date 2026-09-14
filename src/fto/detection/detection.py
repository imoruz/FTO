from contextvars import ContextVar
import time

from dataclasses import dataclass, field
from typing import Callable
from fto.detection.atp import ATPStatus
from fto.detection.restart_config import RestartConfig


@dataclass
class Detection:
    atp: ATPStatus
    site: str                       # 'pre' | 'llm' | 'tool' | 'node' | 'post'
    detail: str = ''
    node_id: str | None = None
    node_seq: int | None = None     # per-node ordinal (every node)
    idx_step: int | None = None     # agent-step index (agents only; injection corr.)
    exception: BaseException | None = None


class KillNode(BaseException):
    """Control-flow node killing signal for IMMEDIATE restart timing.
    """

    def __init__(self, detection: Detection) -> None:
        super().__init__()
        self.detection = detection


@dataclass
class _Scope:
    node_id: str
    node_seq: int
    idx_step: int | None = None
    # If True, a restartable inner LLM/tool detection interrupts the node immediately (Observer in IMMEDIATE restart-timing mode); when False, after node completion
    eager: bool = False
    detections: list[Detection] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)


_active_scope: ContextVar[_Scope | None] = ContextVar('_fto_obs_scope', default=None)

# Optional side-channel that mirrors each recorded Detection elsewhere (eg. OTel span)
_emitter: Callable[[Detection], None] | None = None


def set_detection_emitter(fn: Callable[[Detection], None] | None) -> None:
    global _emitter
    _emitter = fn


class DetectionSink:
    """Where inner probes deposit Detections for the node in flight.
    """

    @staticmethod
    def begin(node_id: str, node_seq: int, idx_step: int | None = None,
              eager: bool = False):
        return _active_scope.set(
            _Scope(node_id, node_seq, idx_step, eager=eager)
        )

    @staticmethod
    def end(token) -> list[Detection]:
        scope = _active_scope.get()
        _active_scope.reset(token)
        return list(scope.detections) if scope else []

    @staticmethod
    def record(detection: Detection) -> None:
        scope = _active_scope.get()
        if scope is None:
            return
        # Enrich detection
        detection.node_id = detection.node_id or scope.node_id
        if detection.node_seq is None:
            detection.node_seq = scope.node_seq
        if detection.idx_step is None:
            detection.idx_step = scope.idx_step
        scope.detections.append(detection)
        # Emit detection
        if _emitter is not None:
            try:
                _emitter(detection)
            except Exception:
                pass
        # IMMEDIATE restart timing: interrupt the node right here for the inner sites
        if scope.eager and detection.site in ('llm', 'tool') \
                and RestartConfig.instance().restartable(
                    detection.atp.code, detection.site, detection.atp.family, detection.atp.recovery_hint):
            raise KillNode(detection)