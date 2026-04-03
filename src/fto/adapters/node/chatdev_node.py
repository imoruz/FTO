from typing import List
from fto.adapters.node.node import NodeAdapter


class ChatDevNodeAdapter(NodeAdapter):
    def __init__(self, inner):
        super().__init__(inner)

    @property
    def id(self) -> str:
        return self._inner.id

    @property
    def input(self):
        return self._inner.input

    @input.setter
    def set_input(self, value):
        self._inner.input = value

    @property
    def is_agent(self) -> bool:
        return self._inner.type == 'agent'

    def append_to_last_message(node, text: str) -> bool:
        from entity.messages import MessageBlock
        if not node.input:
            return False
        last = node.input[-1]
        content = last.content

        if isinstance(content, str):
            new_content = content + "\n\n" + text
        elif isinstance(content, list) and content:
            if isinstance(content[0], MessageBlock):
                new_content = list(content) + [MessageBlock.text_block(text)]
            elif isinstance(content[0], dict):
                new_content = list(content) + [{"type": "text", "text": text}]
            else:
                return False
        else:
            return False
        node.input[-1] = last.with_content(new_content)
        return True
