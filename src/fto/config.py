from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Callable
from fto.adapters.node.node import NodeAdapter
from fto.const import DEFAULT_INJECTED_PROMPT


class FaultType(StrEnum):
    PROMPT_INJECTION=auto()


class RestartMode(StrEnum):
    NO_HISTORY = auto()
    ALL_HISTORY = auto()
    REFINED_HISTORY = auto()


class CircuitState(StrEnum):
    OPEN = auto()
    CLOSED = auto()


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

    def __init__(self, node_id: str, prompt: str = None, mutator: Callable = None, mutator_payload: dict[str, Any] = None):
        super().__init__(node_id)
        self.prompt = prompt or DEFAULT_INJECTED_PROMPT

    @property
    def mode(self):
        return FaultType.PROMPT_INJECTION

    def apply(self, node: NodeAdapter = None):
        if not self.applied:
            self.applied = True
            return node.append_to_last_message(self.prompt)


@dataclass
class FTOConfig:

    node_adapter: NodeAdapter
    patch_target: Any
    patch_method: Any
    fault: Fault = None
    restart_mode: RestartMode = None
    get_edge_propagator: Callable[[Any], Any] = None
    suppress_edge_propagator: Callable[[Any], None] = None
    restore_edge_propagator: Callable[[Any, Any], None] = None
    logger: Any = None

    def resolve_patch_target(self):
        if self.patch_target:
            return self.patch_target
