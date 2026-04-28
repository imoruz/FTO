from copy import copy

from fto.adapters.node.node import NodeAdapter
from fto.faults import Fault
from fto.recovery import Restart
from fto.recovery.checkpoint import Checkpoint


class Manager:
    def __init__(self, fault: Fault | None, restart: Restart | None,
                 node_adapter=lambda node: node,
                 get_edge_propagator=lambda instance: instance._process_edge_output,
                 suppress_edge_propagator=lambda instance: setattr(instance, "_process_edge_output", lambda *a, **kw: None),
                 restore_edge_propagator=lambda instance, original: setattr(instance, "_process_edge_output", original),
                 logger = None, 
                 checkpoint: Checkpoint | None = None):
        self.fault = fault
        self.restart = restart
        self.node_adapter = node_adapter
        self.get_edge_propagator = get_edge_propagator
        self.suppress_edge_propagator = suppress_edge_propagator
        self.restore_edge_propagator = restore_edge_propagator
        self.logger = logger
        self.checkpoint = checkpoint

    def set_node_input(self, node_input):
        self.node_input = node_input

    def _fault_exec(self, exec, callback = None):

        def _exe(instance, node):
            node_adapter: NodeAdapter = self.node_adapter(node)
            if not node_adapter.is_agent:
                exec(instance, node)
                return

            if not self.fault:
                self.logger.log(f"No fault injected. Running original flow.")
                exec(instance, node)
                return
            if node_adapter.id != self.fault.node_id:
                exec(instance, node)
                return
            
            # Checkpoint before agent node executes so we can rollback here on restart
            if self.checkpoint:
                self.checkpoint.save(node_id=node_adapter.id)
                self.logger.log(f"Created checkpoint for {node_adapter.id}.", instance=instance, node_id=node_adapter.id)
            
            if self.restart:
                self.restart.set_context(node_adapter.input)
                
            if not self.fault.applied:
                self.fault.apply(node=node_adapter)
                self.logger.log(f"Fault {self.fault.mode} applied on agent node {node_adapter.id}.", instance=instance, node_id=node_adapter.id)

            # suppress edge propagation
            original = self.get_edge_propagator(instance)
            self.suppress_edge_propagator(instance)

            try:
                exec(instance, node)
                self.logger.log("Faulty execution completed.", instance=instance, node_id=node_adapter.id)
            except Exception as e:
                if not self.fault.raises_on_fault:
                    raise
                self.logger.log(f"Faulty execution raised: {e.__repr__()}", instance=instance, node_id=node_adapter.id)
            finally:
                # restore edge propagation
                self.restore_edge_propagator(instance, original)
            if hasattr(self.fault, "disable"):
                # OTel fault specific
                self.fault.disable()

            callback and callback(node_adapter, instance)

        return _exe

    def _restart_exec(self, exec, on_complete):

        def _exe(instance, node):
            node_adapter: NodeAdapter = self.node_adapter(node)
            if not self.restart:
                self.logger.log("No restart mode configured. Continuing original flow.", instance=instance, node_id=node_adapter.id)
                on_complete()
                return
            
            # Undo changes made by faulty node before restarting
            if self.checkpoint:
                self.logger.log("Undoing changes made by faulty node.")
                self.checkpoint.restore(node_id=node_adapter.id)
            
            self.logger.log(f"Restarting node {node_adapter.id} with mode: {type(self.restart).__name__}")
            new_input = self.restart.get_context()
            self.logger.log(f"New input: {new_input}.")
            node_adapter.set_input(new_input)
            
            exec(instance, node)

            on_complete()

        return _exe
