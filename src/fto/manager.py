from typing import Any, Callable

from fto.adapters.node.node import NodeAdapter
from fto.adapters.edge import EdgeSuppressor
from fto.faults import Fault
from fto.instrumentation import record_injection, record_detection
from fto.detection.observer import Observer, RestartTiming
from fto.recovery import Restart
from fto.recovery.checkpoint import Checkpoint


class Manager:
    def __init__(
        self,
        fault: Fault | None,
        restart: Restart | None,
        edge_suppressor: EdgeSuppressor,
        logger=None,
        checkpoint: Checkpoint | None = None,
        observer: Observer | None = None,
    ) -> None:
        self.fault = fault
        self.restart = restart
        self.edge_suppressor = edge_suppressor
        self.logger = logger
        self.checkpoint = checkpoint
        self.observer = observer
        self.idx_step = 0
        self.node_seq = 0
    
    def snapshot(self, adapter: NodeAdapter):
        if self.checkpoint:
            self.checkpoint.save(node_id=adapter.id)
        if self.restart:
            self.restart.set_context(adapter.input)

    def apply_fault(self, adapter: NodeAdapter):
        self.fault.apply(node=adapter)
        record_injection(
            node_id=adapter.id,
            mode=self.fault.mode,
            idx_step=self.idx_step,
        )
        self.logger.info(
            f'Fault {self.fault.mode} applied on {adapter.id}. New node input: {adapter.input}',
            node_id=adapter.id,
        )

    def _fault_due(self) -> bool:
        return (
            self.fault and self.fault.idx_step == self.idx_step and not self.fault.applied
        )

    def _run(self, callable: Callable[..., Any], fault_applied: bool):
        result = callable()
        if fault_applied and hasattr(self.fault, 'disable'):
            self.fault.disable()
        return result

    def _suppress_edge(self, *args, **kwargs):
        return self.edge_suppressor.suppress(*args, **kwargs) if self.restart else None

    def _restore_edge(self, token, *args, **kwargs):
        if token is not None:
            self.edge_suppressor.restore(token, *args, **kwargs)

    def make_supervised_callable(self, callable: Callable[..., Any], make_adapter: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args, **kwargs):
            adapter: NodeAdapter = make_adapter(*args, **kwargs)
            if not adapter.is_agent:
                return callable(*args, **kwargs)
            if not self.observer:
                return self._run_injection_only(callable, adapter, *args, **kwargs)
            return self._run_observed(callable, adapter, *args, **kwargs)
        return wrapped

    def _run_injection_only(self, callable: Callable[..., Any], adapter: NodeAdapter, *args, **kwargs):
        """No Observer: only an injected fault can trigger recovery."""
        self.idx_step += 1
        if not self._fault_due():
            return callable(*args, **kwargs)

        # snapshot pre-fault exec state, then inject the fault.
        self.snapshot(adapter=adapter)
        self.apply_fault(adapter=adapter)

        token = self._suppress_edge(*args, **kwargs)
        result = self._run(lambda: callable(*args, **kwargs), True)
        self.logger.info('FTO: Faulty execution completed.', node_id=adapter.id)

        if self.restart:
            result = self._restart(callable, adapter, *args, **kwargs)
        self._restore_edge(token, *args, **kwargs)
        return result

    def _run_observed(self, callable: Callable[..., Any], adapter: NodeAdapter, *args, **kwargs):
        self.idx_step += 1
        fault_applied = False

        self.snapshot(adapter=adapter)
        if self._fault_due():
            self.apply_fault(adapter=adapter)
            fault_applied = True

        token = self._suppress_edge(*args, **kwargs)

        result, faults = self._observe(
            callable, adapter, *args, fault_applied=fault_applied, **kwargs
        )
        if faults:
            self._report_faults(faults, adapter)

        restartable = fault_applied or (
            faults and self.observer.should_restart(faults)
        )
        if self.restart and restartable:
            result = self._restart(callable, adapter, *args, **kwargs)

        self._restore_edge(token, *args, **kwargs)
        return result

    def make_supervised_callable_0(
        self, callable: Callable[..., Any], make_adapter: Callable[..., Any]
    ) -> Callable[..., Any]:
        def wrapped(*args, **kwargs):
            adapter: NodeAdapter = make_adapter(*args, **kwargs)

            if not adapter.is_agent:
                return callable(*args, **kwargs)

            self.idx_step += 1
            if not self.fault or not self.fault.idx_step == self.idx_step:
                return callable(*args, **kwargs)

            if self.fault.applied:
                return callable(*args, **kwargs)

            if self.checkpoint:
                self.checkpoint.save(node_id=adapter.id)
            if self.restart:
                self.restart.set_context(adapter.input)

            if not self.fault.applied and self.fault.idx_step == self.idx_step:
                self.fault.apply(node=adapter)
                record_injection(
                    node_id=adapter.id,
                    mode=self.fault.mode,
                    idx_step=self.idx_step,
                )
                self.logger.info(
                    f'Fault {self.fault.mode} applied on {adapter.id}. New node input: {adapter.input}',
                    node_id=adapter.id,
                )

            token = self.edge_suppressor.suppress(*args, **kwargs) if self.restart else None


            faulty_result = callable(*args, **kwargs)
            self.logger.info('Faulty execution completed.', node_id=adapter.id)

            if token is not None:
                self.edge_suppressor.restore(token, *args, **kwargs)

            # Specific to llmmas-otel fault injection
            if hasattr(self.fault, 'disable'):
                self.fault.disable()

            if not self.restart:
                return faulty_result

            if self.checkpoint:
                self.checkpoint.restore(node_id=adapter.id)

            new_input = self.restart.get_context()
            adapter.set_input(new_input)
            self.logger.info(
                f'Restarting {adapter.id} with {type(self.restart).__name__}.',
                node_id=adapter.id,
            )
            return callable(*args, **kwargs)

        return wrapped

    def _observe(self, callable: Callable[..., Any], adapter: NodeAdapter,
                 *args, fault_applied: bool = False, allow_eager=True, **kwargs):
        """One observed pass over the node. Opens a fresh detection scope so the
        inner probes, boundary inspection and verdict all apply to this execution"""
        self.node_seq += 1
        step = self.idx_step if adapter.is_agent else None
        scope = self.observer.begin_scope(
            adapter=adapter, node_seq=self.node_seq, idx_step=step,
            can_restart=self.restart is not None and allow_eager,
        )

        result = self._run(
            lambda: self.observer.run(callable, adapter, *args, **kwargs),
            fault_applied,
        )
        self.logger.info('FTO: Ran observer')
        if self.observer.record_injected_faults and fault_applied:
            self.logger.info(f'FTO: recording injected fault at idx: {self.idx_step} ')
            self.observer._record_injected_fault(idx_step=self.idx_step, node_id=adapter.id, fault=self.fault)
        # post = self.observer.inspect(adapter, 'post')
        detections = self.observer.end_scope(scope)
        self.logger.info(
            f'Observer: {len(detections)} inner detection(s): '
            f'{[(d.atp.code, d.site) for d in detections]}',
            node_id=adapter.id,
        )
        return result, detections

    def _report_faults(self, faults: list, adapter: NodeAdapter) -> None:
        """Report *every* detection for this node"""
        step = self.idx_step if adapter.is_agent else None
        total = len(faults)
        for i, fault in enumerate(faults, start=1):
            self.logger.info(
                f'Observer: ATP {fault.atp.code} ({fault.atp.name}) on {adapter.id} '
                f'[{fault.atp.recovery_hint}] via {fault.site}; '
                f'detection {i}/{total}.',
                node_id=adapter.id,
            )
            record_detection(
                node_id=adapter.id,
                code=fault.atp.code,
                recovery_hint=str(fault.atp.recovery_hint),
                idx_step=step,
            )

    def _restart(self, callable, adapter, *args, **kwargs):
        """Re-execute the node, up to ``restart.restart_count`` times for THIS
        node invocation."""
        max_restarts = getattr(self.restart, 'restart_count', 1)
        result = None
        for attempt in range(1, max_restarts + 1):
            if self.checkpoint:
                self.checkpoint.restore(node_id=adapter.id)
            adapter.set_input(self.restart.get_context())
            self.logger.info(
                f'Restarting {adapter.id} with {type(self.restart).__name__} '
                f'(attempt {attempt}/{max_restarts}).',
                node_id=adapter.id,
            )
            if not self.observer:
                return callable(*args, **kwargs)

            # Observer set: re-run under observation and re-evaluate.
            allow_eager = not (attempt == max_restarts)
            result, faults = self._observe(callable, adapter, allow_eager=allow_eager, *args, **kwargs)
            if faults:
                self._report_faults(faults, adapter)

            if not faults or not self.observer.should_restart(faults):
                return result
            self.logger.info(
                f'{adapter.id} still faulty after restart {attempt}/{max_restarts}.',
                node_id=adapter.id,
            )

        self.logger.info(
            f'Restart budget ({max_restarts}) exhausted for {adapter.id}; '
            f'continuing execution flow.',
            node_id=adapter.id,
        )
        return result