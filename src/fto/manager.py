from copy import copy

from fto.adapters.node.node import NodeAdapter
from fto.config import Fault, RestartMode


class Manager:
    def __init__(self, fault: Fault, restart_mode: RestartMode,
                 node_adapter=lambda node: node,
                 get_edge_propagator=lambda instance: instance._process_edge_output,
                 suppress_edge_propagator=lambda instance: setattr(instance, "_process_edge_output", lambda *a, **kw: None),
                 restore_edge_propagator=lambda instance, original: setattr(instance, "_process_edge_output", original),
                 logger = None):
        self.fault = fault
        self.restart_mode = restart_mode
        self.node_adapter = node_adapter
        self.get_edge_propagator = get_edge_propagator
        self.suppress_edge_propagator = suppress_edge_propagator
        self.restore_edge_propagator = restore_edge_propagator
        self.logger = logger
        self.node_input = None

    def set_node_input(self, node_input):
        self.node_input = node_input

    def _fault_exec(self, exec, callback = None):

        def _exe(instance, node):
            node_adapter: NodeAdapter = self.node_adapter(node)
            if not node_adapter.is_agent:
                exec(instance, node)
                return
            if not self.fault or node_adapter.id != self.fault.node_id:
                exec(instance, node)
                return
            self.set_node_input(node_adapter.input)
            if not self.fault.applied:
                self.fault.apply(node=node_adapter)
                self.logger.log(f"Fault {self.fault.mode} applied on agent node {node.id}.", instance=instance, node_id=node.id)

            # suppress edge propagation
            original = self.get_edge_propagator(instance)
            self.suppress_edge_propagator(instance)

            exec(instance, node)
            self.logger.log("Faulty execution completed.", instance=instance, node_id=node.id)
            # restore edge propagation
            self.restore_edge_propagator(instance, original)

            callback and callback(node, instance)

        return _exe

    def _restart_exec(self, exec, on_complete):

        def _exe(instance, node):
            node: NodeAdapter = self.node_adapter(node)
            if not self.restart_mode:
                self.logger.log("No restart mode configured. Continuing original flow.", instance=instance, node_id=node.id)
                on_complete()
                return
            elif self.restart_mode == RestartMode.ALL_HISTORY:
                self.logger.log(f"Restarting node {node.id} with all history.", instance=instance, node_id=node.id)
                node.input = self.node_input
                exec(instance, node)
            elif self.restart_mode == RestartMode.NO_HISTORY:
                self.logger.log(f"Restarting node {node.id} with no history.", instance=instance, node_id=node.id)
                node.input = []
                exec(instance, node)
            else:
                self.logger.log(f"Restarting node {node.id} with refined history.", instance=instance, node_id=node.id)
                exec(instance, node)

            on_complete()

        return _exe
