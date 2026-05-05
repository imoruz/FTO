from copy import copy

from fto.adapters.node.node import NodeAdapter
from fto.edge import EdgeSuppressor
from fto.faults import Fault
from fto.recovery import Restart
from fto.recovery.checkpoint import Checkpoint


class Manager:
    def __init__(self, fault: Fault | None, restart: Restart | None,
                 edge_suppressor: EdgeSuppressor,
                 logger = None, 
                 checkpoint: Checkpoint | None = None):
        self.fault = fault
        self.restart = restart
        self.edge_suppressor = edge_suppressor
        self.logger = logger
        self.checkpoint = checkpoint

    def set_node_input(self, node_input):
        self.node_input = node_input

    def make_supervised_callable(self, callable, make_adapter):
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
                self.logger.info(f"Fault {self.fault.mode} applied on {adapter.id}. New node input: {adapter.input}", node_id=adapter.id)

            token = self.edge_suppressor.suppress(*args, **kwargs)
            try:
                callable(*args, **kwargs)   # faulty run; result intentionally discarded
                self.logger.info("Faulty execution completed.", node_id=adapter.id)
            except Exception as e:
                if not self.fault.raises_on_fault:
                    raise
                self.logger.info(f"Faulty execution raised: {e!r}", node_id=adapter.id)
            finally:
                self.edge_suppressor.restore(token, *args, **kwargs)

            if hasattr(self.fault, "disable"):
                self.fault.disable()

            if not self.restart:
                return None

            if self.checkpoint:
                self.checkpoint.restore(node_id=adapter.id)

            new_input = self.restart.get_context()
            adapter.set_input(new_input)
            self.logger.info(f"Restarting {adapter.id} with {type(self.restart).__name__}.", node_id=adapter.id)
            return callable(*args, **kwargs)

        return wrapped

    # def _fault_exec(self, exec, callback = None):

    #     def _exe(instance, node):
    #         node_adapter: NodeAdapter = self.node_adapter(node)
    #         if not node_adapter.is_agent:
    #             exec(instance, node)
    #             return

    #         if not self.fault:
    #             self.logger.info(f"No fault injected. Running original flow.")
    #             exec(instance, node)
    #             return
    #         if node_adapter.id != self.fault.node_id:
    #             exec(instance, node)
    #             return
            
    #         # Checkpoint before agent node executes so we can rollback here on restart
    #         if self.checkpoint:
    #             self.checkpoint.save(node_id=node_adapter.id)
    #             self.logger.info(f"Created checkpoint for {node_adapter.id}.", node_id=node_adapter.id)
            
    #         if self.restart:
    #             self.restart.set_context(node_adapter.input)
                
    #         if not self.fault.applied:
    #             self.fault.apply(node=node_adapter)
    #             self.logger.info(f"Fault {self.fault.mode} applied on agent node {node_adapter.id}.", node_id=node_adapter.id)

    #         # suppress edge propagation
    #         original = self.get_edge_propagator(instance)
    #         self.suppress_edge_propagator(instance)

    #         try:
    #             exec(instance, node)
    #             self.logger.info("Faulty execution completed.", node_id=node_adapter.id)
    #         except Exception as e:
    #             if not self.fault.raises_on_fault:
    #                 raise
    #             self.logger.info(f"Faulty execution raised: {e.__repr__()}", node_id=node_adapter.id)
    #         finally:
    #             # restore edge propagation
    #             self.restore_edge_propagator(instance, original)
    #         if hasattr(self.fault, "disable"):
    #             # OTel fault specific
    #             self.fault.disable()

    #         callback and callback(node_adapter, instance)

    #     return _exe

    # def _restart_exec(self, exec, on_complete):

    #     def _exe(instance, node):
    #         node_adapter: NodeAdapter = self.node_adapter(node)
    #         if not self.restart:
    #             self.logger.info("No restart mode configured. Continuing original flow.", node_id=node_adapter.id)
    #             on_complete()
    #             return
            
    #         # Undo changes made by faulty node before restarting
    #         if self.checkpoint:
    #             self.logger.info("Undoing changes made by faulty node.")
    #             self.checkpoint.restore(node_id=node_adapter.id)
            
    #         self.logger.info(f"Restarting node {node_adapter.id} with mode: {type(self.restart).__name__}")
    #         new_input = self.restart.get_context()
    #         self.logger.info(f"New input: {new_input}.")
    #         node_adapter.set_input(new_input)
            
    #         exec(instance, node)

    #         on_complete()

    #     return _exe
