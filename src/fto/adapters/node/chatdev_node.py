from typing import Any, List

from fto.adapters.node.node import NodeAdapter, content_text, replace_content_text
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

    def context_as_list(self, context: Any = None) -> List[str]:
        messages = self.input if context is None else context
        return [_message_text(message) for message in messages or []]

    def context_from_list(self, texts: List[str], context: Any = None) -> Any:
        messages = list((self.input if context is None else context) or [])
        if len(texts) != len(messages):
            raise ValueError(
                f'context_from_list got {len(texts)} text entries for '
                f'{len(messages)} messages; they must line up one to one.'
            )
        return [
            _message_with_text(message, text) if text else message
            for message, text in zip(messages, texts)
        ]


def _message_text(message: Any) -> str:
    # A tool call or its result is half of a pair the provider matches up;
    # rewriting either side would break that, so report no text.
    if getattr(message, 'tool_calls', None) or getattr(message, 'tool_call_id', None):
        return ''
    return content_text(message.content)


def _message_with_text(message: Any, text: str) -> Any:
    content = replace_content_text(message.content, text, _text_block_like)
    if content is None:
        return message
    return message.with_content(content)


def _text_block_like(block: Any, text: str) -> Any:
    if isinstance(block, dict):
        return {'type': 'text', 'text': text}
    from entity.messages import MessageBlock

    return MessageBlock.text_block(text)
