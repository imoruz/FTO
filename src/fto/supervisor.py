from fto.config import FTOConfig, CircuitState
from fto.manager import Manager

class Supervisor:
    def __init__(self, fto_config: FTOConfig):
        self.manager = Manager(
            fault=fto_config.fault,
            restart=fto_config.restart,
            edge_suppressor=fto_config.edge_suppressor,
            logger=fto_config.logger,
            checkpoint=fto_config.checkpoint
        )
        self.checkpoint = fto_config.checkpoint
        self.instrumentation = fto_config.instrumentation


        self.logger = fto_config.logger

    # def _patch(self, fn):
    #     setattr(self.patch_target, self.patch_method, fn)

    # def _restore(self):
    #     self._patch(self._original_node_exec)

    def start(self, target, function_name, make_adapter):
        self.logger.info("Started supervisor.")
        # apply framework-specific otel patches
        if self.instrumentation:
            self.instrumentation()
            # recapture so the fault executor calls the OTEL patch
        original = getattr(target, function_name)
        if self.checkpoint:
            self.checkpoint.save_baseline()
        wrapped = self.manager.make_supervised_callable(original, make_adapter)
        setattr(target, function_name, wrapped)
        # self._patch(
        #     self.manager._fault_exec(self._original_node_exec, self._on_fault)
        # )

    # def _on_fault(self, node, instance):
    #     self.logger.info("Supervisor received fault signal.", instance=instance, node_id=node.id)
    #     if self.node_states.get(node.id) == CircuitState.OPEN:
    #         return
    #     self.node_states[node.id] = CircuitState.OPEN

    #     self._patch(
    #         self.manager._restart_exec(self._original_node_exec, self._restore)
    #     )
    #     # restarting
    #     getattr(self.patch_target, self.patch_method)(instance, node._inner)
