from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class NodeProtocol(Protocol):

    @property
    def id(self) -> str: ...

    @property
    def node_type(self) -> str: ...

    @property
    def input(self) -> list: ...

    @property
    def last_message(self) -> str: ...

    @input.setter
    def input(self, value: list) -> None: ...

    @property
    def is_agent(self) -> bool: ...
