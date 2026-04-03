from fto.config import FTOConfig, CircuitState
from fto.manager import Manager

class Supervisor:
    def __init__(self, node_ids: list[str], fto_config: FTOConfig):
        self.manager = Manager(
            fault=fto_config.fault,
            restart_mode=fto_config.restart_mode,
            node_adapter=fto_config.node_adapter,
            get_edge_propagator=fto_config.get_edge_propagator,
            suppress_edge_propagator=fto_config.suppress_edge_propagator,
            restore_edge_propagator=fto_config.restore_edge_propagator,
            logger=fto_config.logger
        )
        self.node_states = {node_id: CircuitState.CLOSED for node_id in node_ids}

        self.patch_target = fto_config.patch_target
        self.patch_method = fto_config.patch_method
        self._original_node_exec = getattr(self.patch_target, self.patch_method)

        self.logger = fto_config.logger

    def _patch(self, fn):
        setattr(self.patch_target, self.patch_method, fn)

    def _restore(self):
        self._patch(self._original_node_exec)

    def start(self):
        self.logger.log("Started supervisor.")
        self._patch(
            self.manager._fault_exec(self._original_node_exec, self._on_fault)
        )

    def _on_fault(self, node, instance):
        self.logger.log("Supervisor received fault signal.", instance=instance, node_id=node.id)
        if self.node_states.get(node.id) == CircuitState.OPEN:
            return
        self.node_states[node.id] == CircuitState.OPEN

        self._patch(
            self.manager._restart_exec(self._original_node_exec, self._restore)
        )
        # restarting
        getattr(self.patch_target, self.patch_method)(instance, node)
