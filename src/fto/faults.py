from enum import Enum, StrEnum, auto

from aegis_mas.aegis_core import FMMaliciousFactory, FMErrorType
from llmmas_otel.injection import (
    enable_fault_injection,
    disable_fault_injection,
    SpecFaultEngine,
    FaultSpec,
)
from ollama import Client

from fto.adapters.node.node import NodeAdapter
from fto.const import DEFAULT_INJECTED_PROMPT
from fto.utils import LLMAdapter


class FaultType(StrEnum):
    PROMPT_INJECTION = auto()
    OTEL_INFRA = auto()


class Fault:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.raises_on_fault = False
        self.applied = False

    @property
    def mode(self) -> Enum:
        pass

    def apply(self) -> None:
        self.applied = True


class PromptInjectionFault(Fault):
    def __init__(self, node_id: str, prompt: str = None) -> None:
        super().__init__(node_id)
        self.prompt = prompt or DEFAULT_INJECTED_PROMPT

    @property
    def mode(self) -> FaultType:
        return FaultType.PROMPT_INJECTION

    def apply(self, node: NodeAdapter) -> None:
        if self.applied:
            return
        self.applied = True
        node.append_to_last_message(self.prompt)


class AegisFault(Fault):
    def __init__(
        self,
        node_id,
        mode: FMErrorType,
        llm_provider: str = 'ollama',
        llm_model: str = 'mistral',
    ) -> None:
        super().__init__(node_id)
        self.factory = FMMaliciousFactory(
            llm=LLMAdapter(
                client=Client(host='http://localhost:11434'), model=llm_model
            )
        )
        self.fm_mode = mode
        self.agent_context = None

    @property
    def mode(self) -> FMErrorType:
        return self.fm_mode

    def apply(self, node: NodeAdapter) -> None:
        if self.applied:
            return
        self.applied = True
        original_last_message = node.last_message
        corrupted_last_message = self.factory.inject_prompt(
            prompt=original_last_message,
            fm_error_type=self.mode,
            agent_context=node.to_aegis_context(),
        )
        node.overwrite_last_message(text=corrupted_last_message)


class OTelFault(Fault):
    def __init__(self, node_id: str, specs: list[dict], seed: str = 'default'):
        super().__init__(node_id)
        self.specs = specs
        self.seed = seed
        self.raises_on_fault = True

    @property
    def mode(self) -> FaultType:
        return FaultType.OTEL_INFRA

    def apply(self, node: NodeAdapter) -> None:
        if self.applied:
            return
        self.applied = True
        specs_with_selector = []
        for d in self.specs:
            if d.get('hook') == 'llm_call':
                specs_with_selector.append(d)
            else:
                specs_with_selector.append(
                    {
                        **d,
                        'selector': {
                            **d.get('selector', {}),
                            'source_agent_id': self.node_id,
                        },
                    }
                )
        parsed = [FaultSpec.from_dict(d) for d in specs_with_selector]
        enable_fault_injection(SpecFaultEngine(specs=parsed, seed=self.seed))

    def disable(self) -> None:
        disable_fault_injection()
