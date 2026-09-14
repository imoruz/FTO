from typing import Any

from fto.adapters.node.node import NodeAdapter
from aegis_mas.aegis_core import AgentContext


class ChatDevNodeAdapter(NodeAdapter):
    def __init__(self, inner) -> None:
        super().__init__(inner)

    @property
    def id(self) -> str:
        return self._inner.id

    @property
    def input(self) -> Any:
        return self._inner.input

    @property
    def last_message(self) -> Any:
        if not self.input:
            return None
        last_content = self.input[-1].content
        if isinstance(last_content, str):
            return last_content
        return last_content[-1].text

    def set_input(self, value) -> None:
        self._inner.input = value

    @property
    def is_agent(self) -> bool:
        return self._inner.type == 'agent'

    def to_aegis_context(self) -> AgentContext:
        return AgentContext(
            # role_name = self.id,
            role_type=self._inner.type,
            agent_id=self.id,
            system_message=self._inner.role,
            tools=[tool.get('name', '') for tool in self._inner.tools],
            # external_tools = [],
            description=self._inner.description,
            model_type=self._inner.model_name,
            recent_history=[msg.content for msg in self._inner.input],
        )

    def append_to_last_message(self, text: str) -> None:
        from entity.messages import MessageBlock

        if not self.input:
            return
        last = self.input[-1]
        content = last.content

        if isinstance(content, str):
            new_content = content + '\n\n' + text
        elif isinstance(content, list) and content:
            if isinstance(content[0], MessageBlock):
                new_content = list(content) + [MessageBlock.text_block(text)]
            elif isinstance(content[0], dict):
                new_content = list(content) + [{'type': 'text', 'text': text}]
            else:
                return
        else:
            return
        self._inner.input[-1] = last.with_content(new_content)

    def overwrite_last_message(self, text: str) -> None:
        from entity.messages import MessageBlock

        if not self.input:
            return
        last = self.input[-1]
        content = last.content

        if isinstance(content, str):
            new_content = text
        elif isinstance(content, list) and content:
            if isinstance(content[0], MessageBlock):
                new_content = [MessageBlock.text_block(text)]
            elif isinstance(content[0], dict):
                new_content = [{'type': 'text', 'text': text}]
            else:
                return
        else:
            return
        self._inner.input.pop()
        self._inner.input.append(last.with_content(new_content))
