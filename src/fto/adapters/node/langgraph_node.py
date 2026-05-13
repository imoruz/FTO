from typing import Any

from aegis_mas.aegis_core import AgentContext

from fto.adapters.node.node import NodeAdapter


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
        return self._task.input.get('messages', [])

    def set_input(self, msgs: Any) -> None:
        self._task.input['messages'] = msgs

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
        msgs = self._task.input['messages']
        if msgs:
            from langchain_core.messages import AIMessage

            msgs.append(AIMessage(content=text))

    def overwrite_last_message(self, text: str) -> None:
        msgs = self._task.input['messages']
        if msgs:
            msgs[-1] = msgs[-1].copy(update={'content': text})
