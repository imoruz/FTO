from fto.adapters.node.langgraph_node import LangGraphNodeAdapter


class FakeMessage:
    def __init__(self, content):
        self.content = content

    def copy(self, update=None):
        new_content = self.content
        if update and 'content' in update:
            new_content = update['content']
        return FakeMessage(new_content)


class FakeInput:
    def __init__(self, messages):
        self.messages = messages


class FakeTask:
    def __init__(self, name, messages):
        self.name = name
        self.input = FakeInput(messages)


NODE_SPEC = {
    'type': 'agent',
    'role': 'a helpful assistant',
    'tools': ['search'],
    'description': 'does things',
    'model': 'gpt-4',
}


class TestLangGraphNodeAdapter:
    def test_id_returns_task_name(self):
        task = FakeTask('node-1', [])
        adapter = LangGraphNodeAdapter(task, NODE_SPEC)
        assert adapter.id == 'node-1'

    def test_is_agent_reflects_spec_type(self):
        task = FakeTask('node-1', [])
        assert LangGraphNodeAdapter(task, {'type': 'agent'}).is_agent is True
        assert LangGraphNodeAdapter(task, {'type': 'tool'}).is_agent is False
        assert LangGraphNodeAdapter(task, {}).is_agent is False

    def test_input_returns_task_messages(self):
        messages = [FakeMessage('hi')]
        task = FakeTask('node-1', messages)
        adapter = LangGraphNodeAdapter(task, NODE_SPEC)
        assert adapter.input is messages

    def test_set_input_replaces_task_messages(self):
        task = FakeTask('node-1', [FakeMessage('old')])
        adapter = LangGraphNodeAdapter(task, NODE_SPEC)
        new_messages = [FakeMessage('new')]

        adapter.set_input(new_messages)

        assert task.input.messages is new_messages

    def test_last_message_returns_content_of_last_message(self):
        task = FakeTask('node-1', [FakeMessage('first'), FakeMessage('last')])
        adapter = LangGraphNodeAdapter(task, NODE_SPEC)
        assert adapter.last_message == 'last'

    def test_last_message_is_none_when_no_messages(self):
        task = FakeTask('node-1', [])
        adapter = LangGraphNodeAdapter(task, NODE_SPEC)
        assert adapter.last_message is None

    def test_to_aegis_context_maps_spec_and_history(self):
        messages = [FakeMessage('hi'), FakeMessage('there')]
        task = FakeTask('node-1', messages)
        adapter = LangGraphNodeAdapter(task, NODE_SPEC)

        ctx = adapter.to_aegis_context()

        assert ctx.role_type == 'agent'
        assert ctx.agent_id == 'node-1'
        assert ctx.system_message == 'a helpful assistant'
        assert ctx.tools == ['search']
        assert ctx.description == 'does things'
        assert ctx.model_type == 'gpt-4'
        assert ctx.recent_history == ['hi', 'there']

    def test_append_to_last_message_adds_an_ai_message(self, fake_module):
        captured = {}

        class FakeAIMessage:
            def __init__(self, content):
                self.content = content
                captured['created'] = self

        fake_module('langchain_core.messages', AIMessage=FakeAIMessage)

        task = FakeTask('node-1', [FakeMessage('hi')])
        adapter = LangGraphNodeAdapter(task, NODE_SPEC)

        adapter.append_to_last_message('appended')

        assert len(task.input.messages) == 2
        assert task.input.messages[-1] is captured['created']
        assert task.input.messages[-1].content == 'appended'

    def test_append_to_last_message_noop_when_no_messages(self, fake_module):
        fake_module('langchain_core.messages', AIMessage=object)

        task = FakeTask('node-1', [])
        adapter = LangGraphNodeAdapter(task, NODE_SPEC)

        adapter.append_to_last_message('appended')

        assert task.input.messages == []

    def test_overwrite_last_message_replaces_content_of_last_message(self):
        first, last = FakeMessage('hi'), FakeMessage('there')
        task = FakeTask('node-1', [first, last])
        adapter = LangGraphNodeAdapter(task, NODE_SPEC)

        adapter.overwrite_last_message('replaced')

        assert task.input.messages[0] is first
        assert task.input.messages[-1] is not last
        assert task.input.messages[-1].content == 'replaced'

    def test_overwrite_last_message_noop_when_no_messages(self):
        task = FakeTask('node-1', [])
        adapter = LangGraphNodeAdapter(task, NODE_SPEC)

        adapter.overwrite_last_message('replaced')

        assert task.input.messages == []
