from typing import Any
from aegis_core import AgentContext


class NodeAdapter:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def id(self) -> str:
        pass

    @property
    def input(self) -> Any:
        pass

    @property
    def last_message(self) -> Any:
        pass

    @property
    def is_agent(self) -> bool:
        pass

    def set_input(self, value: Any) -> None:
        pass

    def to_aegis_context(self) -> AgentContext:
        pass

    def append_to_last_message(self, text: str) -> None:
        pass

    def overwrite_last_message(self, text: str) -> None:
        pass
