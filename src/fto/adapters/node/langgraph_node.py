from typing import Any, List

from aegis_mas.aegis_core import AgentContext

from fto.adapters.node.node import NodeAdapter, content_text, replace_content_text


class LangGraphNodeAdapter(NodeAdapter):
    def __init__(self, task, node_spec: dict) -> None:
        self._task = task
        self._spec = node_spec

    @property
    def id(self) -> str:
        return self._task.name

    @property
    def is_agent(self) -> bool:
        return self._spec.get('type') == 'agent'

    @property
    def input(self) -> Any:
        return self._task.input.messages

    def set_input(self, msgs: Any) -> None:
        self._task.input.messages = msgs

    @property
    def last_message(self) -> Any:
        msgs = self.input
        return msgs[-1].content if msgs else None

    def to_aegis_context(self) -> AgentContext:
        return AgentContext(
            role_type=self._spec.get('type', 'agent'),
            agent_id=self.id,
            system_message=self._spec.get('role', ''),
            tools=self._spec.get('tools', []),
            description=self._spec.get('description', ''),
            model_type=self._spec.get('model', ''),
            recent_history=[m.content for m in self.input],
        )

    def append_to_last_message(self, text: str) -> None:
        msgs = self._task.input.messages
        if msgs:
            from langchain_core.messages import AIMessage

            msgs.append(AIMessage(content=text))

    def overwrite_last_message(self, text: str) -> None:
        msgs = self._task.input.messages
        if msgs:
            msgs[-1] = msgs[-1].copy(update={'content': text})

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
    return message.copy(update={'content': content})


def _text_block_like(block: Any, text: str) -> Any:
    # LangChain carries multimodal content as plain dicts.
    return {'type': 'text', 'text': text}
