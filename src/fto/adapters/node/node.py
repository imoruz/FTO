from dataclasses import dataclass
from typing import Any


class NodeAdapter:
    def __init__(self, inner: Any):
        self._inner = inner

    @property
    def id(self) -> str:
        pass

    @property
    def input(self):
        pass

    @property
    def is_agent(self) -> bool:
        pass

    @input.setter
    def set_input(self, value):
        pass

    def append_to_last_message(self, text: str) -> None:
        pass
