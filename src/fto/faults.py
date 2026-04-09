from enum import StrEnum, auto
from typing import Any, Callable

from aegis_core import AgentContext, FMMaliciousFactory, FMErrorType
from ollama import Client

from fto.adapters.node.node import NodeAdapter
from fto.const import DEFAULT_INJECTED_PROMPT
from fto.utils import LLMAdapter

class FaultType(StrEnum):
    PROMPT_INJECTION=auto()


class Fault:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.applied = False

    @property
    def mode(self):
        pass

    def apply(self):
        self.applied = True


class PromptInjectionFault(Fault):
    def __init__(self, node_id: str, prompt: str = None):
        super().__init__(node_id)
        self.prompt = prompt or DEFAULT_INJECTED_PROMPT

    @property
    def mode(self):
        return FaultType.PROMPT_INJECTION

    def apply(self, node: NodeAdapter):
        if self.applied:
            return
        self.applied = True
        return node.append_to_last_message(self.prompt)
        

class AegisFault(Fault):
    """Class based on AEGIS to inject MAST based faults"""
    def __init__(self, node_id, fm_error_type: FMErrorType):
        super().__init__(node_id)
        self.factory = FMMaliciousFactory(llm=LLMAdapter(client=Client(host="http://localhost:11434")))
        self.fm_error_type = fm_error_type

    def set_agent_context(self, agent_context: AgentContext):
        self.agent_context = agent_context

    def apply(self, node: NodeAdapter):
        if self.applied:
            return
        self.applied = True
        original_last_message = node.last_message
        corrupted_last_message = self.factory.inject_prompt(
            prompt=original_last_message,
            fm_error_type=self.fm_error_type,
            agent_context=self.agent_context
        )
        node.overwrite_last_message(text=corrupted_last_message)
