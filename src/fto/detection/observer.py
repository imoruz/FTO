from __future__ import annotations

import concurrent.futures
import contextvars
import time
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any, Callable, List
from fto.detection.classifier import classify_exception
from fto.detection.detection import Detection, DetectionSink, KillNode
from fto.detection.atp import (
    AEGIS_FM_TO_ATP,
    EXCEPTION_MAPPING,
    OTEL_ACTION_TO_ATP,
    ATPStatus,
    status,
)
from fto.detection.restart_config import RestartConfig
from fto.adapters.node.node import NodeAdapter

from fto.faults import Fault, FaultType


class RestartTiming(StrEnum):
    DEFERRED = auto()
    IMMEDIATE = auto()


class ContextInspector:
    """Boundary-level context-fault detection. Eg: LLM as judge"""

    def __init__(self, signals: list[tuple] | None = None,
                 phases: tuple[str, ...] = ('pre', 'post')):
        pass


    def inspect(self, adapter: NodeAdapter, phase: str) -> Detection | None:
        pass


class Observer:
    """Owns scope lifecycle, boundary inspection, and the hang watchdog.
    """

    def __init__(self, inspector: ContextInspector | None = None,
                 timeout_node: float | None = None,
                 timeout_llm: float | None = None,
                 timeout_tool: float | None = None,
                 logger=None,
                 record_injected_faults: bool = False,
                 restart_timing: RestartTiming | str = RestartTiming.DEFERRED,
                 restart_codes: set[int] | None = None, restart_sites: set[str] | None = None, restart_families: set[str] | None = None):
        self.inspector = inspector if inspector is not None else ContextInspector()
        self.timeout_node = timeout_node
        self.timeout_llm = timeout_llm
        self.timeout_tool = timeout_tool
        self.logger = logger
        # Explicitly record injected faults flag (if False, tries to detect them instead)
        self.record_injected_faults = record_injected_faults
        self.restart_timing = RestartTiming(restart_timing)
        # Shared runtime allow-list, also consulted by the detector (KillNode).
        self.restart_config = RestartConfig.configure(
            codes=restart_codes, sites=restart_sites, families=restart_families
        )

    def begin_scope(self, adapter, node_seq: int, idx_step: int | None = None,
                    can_restart: bool = False):
        eager = self.restart_timing == RestartTiming.IMMEDIATE and can_restart
        return DetectionSink.begin(adapter.id, node_seq, idx_step, eager=eager)


    def _record_injected_fault(self, idx_step: int | None, node_id: str | None, fault: Fault | None) -> None:
        """Oracle ground truth: at the step where FTO injects a (mapped) fault,
        pre-record its ATP code(s) into the just-opened scope."""
        if fault is None or idx_step is None:
            return
        if idx_step != getattr(fault, 'idx_step', None):
            return

        mode = getattr(fault, 'mode', None)
        if mode == FaultType.OTEL_INFRA:
            self._record_injected_otel_faults(idx_step, fault)
            return

        code = AEGIS_FM_TO_ATP.get(mode)
        if code is None:
            return
        DetectionSink.record(
            Detection(status(code), 'node',
                      detail=f'Injected {mode}', idx_step=idx_step)
        )

    def _record_injected_otel_faults(self, idx_step: int | None, fault: Fault | None) -> None:
        for spec in getattr(fault, 'specs', None) or []:
            action_type = (spec.get('action') or {}).get('type')
            code = OTEL_ACTION_TO_ATP.get(action_type)
            if code is None:
                if self.logger:
                    self.logger.info(
                        f'FTO: no ATP mapping for OTel action {action_type!r}; skipping.'
                    )
                continue
            DetectionSink.record(
                Detection(status(code), 'node',
                          detail=f'Injected OTel fault {action_type}', idx_step=idx_step)
            )

    def end_scope(self, token) -> list[Detection]:
        return DetectionSink.end(token)

    def run(self, callable: Callable[..., Any], adapter, *args, **kwargs):
        """Run the node. Inner exceptions are recorded by the probes and
        (usually) swallowed by the framework, so they surface as Detections in
        the active scope rather than as raises here."""
        if self.timeout_node is None:
            try:
                return callable(*args, **kwargs)
            except KillNode:
                # IMMEDIATE timing: a restartable inner fault interrupted the node
                return None
            except KeyboardInterrupt:
                # ctrl + c termination should interrupt the process, not get swallowed
                raise
            except BaseException as exc:
                DetectionSink.record(classify_exception(exc, 'node'))
                return None

        ctx = contextvars.copy_context()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fut = executor.submit(ctx.run, callable, *args, **kwargs)
        try:
            return fut.result(timeout=self.timeout_node)
        except concurrent.futures.TimeoutError:
            executor.shutdown(wait=False)
            DetectionSink.record(
                Detection(status(504), 'node',
                          detail=f'node hang > {self.timeout_node}s')
            )
            return None
        except KillNode:
            executor.shutdown(wait=False)
            return None
        except KeyboardInterrupt:
            executor.shutdown(wait=False)
            raise
        except BaseException as exc:
            executor.shutdown(wait=False)
            DetectionSink.record(classify_exception(exc, 'node'))
            return None

    def should_restart(self, detections: List[Detection]) -> bool:
        if not self.restart_config.configured:
            # Fallback on atp default hint
            return any(d.atp.is_restartable() for d in detections)
        return any(
            self.restart_config.restartable(d.atp.code, d.site, d.atp.family, d.atp.recovery_hint)
            for d in detections
        )
