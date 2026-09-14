from typing import Any, Callable

from fto.config import FTOConfig
from fto.manager import Manager


class Supervisor:
    def __init__(self, fto_config: FTOConfig) -> None:
        self.manager = Manager(
            fault=fto_config.fault,
            restart=fto_config.restart,
            edge_suppressor=fto_config.edge_suppressor,
            logger=fto_config.logger,
            checkpoint=fto_config.checkpoint,
            observer=fto_config.observer,
        )
        self.checkpoint = fto_config.checkpoint
        self.instrumentation = fto_config.instrumentation
        self.observer = fto_config.observer
        self.probe = fto_config.probe

        self.logger = fto_config.logger

    def start(
        self, target: Any, function_name: str, make_adapter: Callable[..., Any]
    ) -> None:
        self.logger.info('Started supervisor.')
        if self.observer is not None and self.probe:
            self.probe.install(
                timeout_llm=self.observer.timeout_llm,
                timeout_tool=self.observer.timeout_tool,
            )
        # apply framework-specific OTel patches (now wrapping the probes)
        if self.instrumentation:
            self.instrumentation()
        # recapture so the fault executor calls the OTel patch or original function if not patched
        original = getattr(target, function_name)
        if self.checkpoint:
            self.checkpoint.save_baseline()
        wrapped = self.manager.make_supervised_callable(original, make_adapter)
        setattr(target, function_name, wrapped)
