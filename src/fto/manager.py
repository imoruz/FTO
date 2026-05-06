from typing import Any, Callable

from fto.adapters.node.node import NodeAdapter
from fto.edge import EdgeSuppressor
from fto.faults import Fault
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
    ) -> None:
        self.fault = fault
        self.restart = restart
        self.edge_suppressor = edge_suppressor
        self.logger = logger
        self.checkpoint = checkpoint

    def make_supervised_callable(
        self, callable: Callable[..., Any], make_adapter: Callable[..., Any]
    ) -> Callable[..., Any]:
        def wrapped(*args, **kwargs):
            adapter: NodeAdapter = make_adapter(*args, **kwargs)

            if not adapter.is_agent:
                return callable(*args, **kwargs)
            if not self.fault or adapter.id != self.fault.node_id:
                return callable(*args, **kwargs)

            if self.checkpoint:
                self.checkpoint.save(node_id=adapter.id)
            if self.restart:
                self.restart.set_context(adapter.input)

            if not self.fault.applied:
                self.fault.apply(node=adapter)
                self.logger.info(
                    f'Fault {self.fault.mode} applied on {adapter.id}. New node input: {adapter.input}',
                    node_id=adapter.id,
                )

            token = self.edge_suppressor.suppress(*args, **kwargs)
            try:
                callable(*args, **kwargs)
                self.logger.info('Faulty execution completed.', node_id=adapter.id)
            except Exception as e:
                if not self.fault.raises_on_fault:
                    raise
                self.logger.info(f'Faulty execution raised: {e!r}', node_id=adapter.id)
            finally:
                self.edge_suppressor.restore(token, *args, **kwargs)

            # Specific to llmmas-otel fault injection
            if hasattr(self.fault, 'disable'):
                self.fault.disable()

            if not self.restart:
                return None

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
