import pytest

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


class FakeToolMessage(FakeMessage):
    def __init__(self, content, tool_calls=None, tool_call_id=None):
        super().__init__(content)
        self.tool_calls = tool_calls or []
        self.tool_call_id = tool_call_id


class TestLangGraphNodeAdapterContextAsList:
    def test_reads_the_text_of_every_message(self):
        task = FakeTask('n', [FakeMessage('the plan'), FakeMessage('the report')])
        adapter = LangGraphNodeAdapter(task, NODE_SPEC)

        assert adapter.context_as_list() == ['the plan', 'the report']

    def test_reads_a_snapshot_instead_of_the_live_input(self):
        adapter = LangGraphNodeAdapter(FakeTask('n', [FakeMessage('live')]), NODE_SPEC)

        assert adapter.context_as_list([FakeMessage('snapshot')]) == ['snapshot']

    def test_joins_the_text_blocks_of_block_list_content(self):
        content = [{'type': 'text', 'text': 'first'}, {'type': 'text', 'text': 'second'}]
        adapter = LangGraphNodeAdapter(FakeTask('n', [FakeMessage(content)]), NODE_SPEC)

        assert adapter.context_as_list() == ['first\n\nsecond']

    def test_non_text_blocks_contribute_no_text(self):
        content = [{'type': 'image_url', 'image_url': 'http://x/y.png'},
                   {'type': 'text', 'text': 'the issue'}]
        adapter = LangGraphNodeAdapter(FakeTask('n', [FakeMessage(content)]), NODE_SPEC)

        assert adapter.context_as_list() == ['the issue']

    def test_tool_protocol_messages_report_an_empty_entry(self):
        messages = [
            FakeToolMessage('calling a tool', tool_calls=[{'id': '1'}]),
            FakeToolMessage('the tool result', tool_call_id='1'),
        ]
        adapter = LangGraphNodeAdapter(FakeTask('n', messages), NODE_SPEC)

        assert adapter.context_as_list() == ['', '']

    def test_an_empty_context_reads_as_an_empty_list(self):
        adapter = LangGraphNodeAdapter(FakeTask('n', []), NODE_SPEC)
        assert adapter.context_as_list() == []


class TestLangGraphNodeAdapterContextFromList:
    def test_replaces_string_content_message_by_message(self):
        task = FakeTask('n', [FakeMessage('the plan'), FakeMessage('the report')])
        adapter = LangGraphNodeAdapter(task, NODE_SPEC)

        rebuilt = adapter.context_from_list(['compressed plan', 'compressed report'])

        assert [m.content for m in rebuilt] == ['compressed plan', 'compressed report']

    def test_an_empty_entry_keeps_that_message_exactly_as_it_was(self):
        kept = FakeMessage('the review')
        task = FakeTask('n', [FakeMessage('the plan'), kept])
        adapter = LangGraphNodeAdapter(task, NODE_SPEC)

        rebuilt = adapter.context_from_list(['compressed plan', ''])

        assert rebuilt[0].content == 'compressed plan'
        assert rebuilt[1] is kept

    def test_it_does_not_touch_the_context_it_was_given(self):
        messages = [FakeMessage('the plan')]
        adapter = LangGraphNodeAdapter(FakeTask('n', messages), NODE_SPEC)

        rebuilt = adapter.context_from_list(['compressed plan'])

        assert messages[0].content == 'the plan'
        assert rebuilt[0] is not messages[0]

    def test_block_list_content_keeps_its_block_shape(self):
        content = [{'type': 'image_url', 'image_url': 'http://x/y.png'},
                   {'type': 'text', 'text': 'the issue'}]
        adapter = LangGraphNodeAdapter(FakeTask('n', [FakeMessage(content)]), NODE_SPEC)

        rebuilt = adapter.context_from_list(['compressed issue'])

        assert rebuilt[0].content == [
            {'type': 'image_url', 'image_url': 'http://x/y.png'},
            {'type': 'text', 'text': 'compressed issue'},
        ]

    def test_a_message_with_no_text_to_replace_is_left_alone(self):
        message = FakeMessage([{'type': 'image_url', 'image_url': 'http://x/y.png'}])
        adapter = LangGraphNodeAdapter(FakeTask('n', [message]), NODE_SPEC)

        assert adapter.context_from_list(['compressed'])[0] is message

    def test_a_mismatched_number_of_entries_is_an_error(self):
        task = FakeTask('n', [FakeMessage('a'), FakeMessage('b')])
        adapter = LangGraphNodeAdapter(task, NODE_SPEC)

        with pytest.raises(ValueError, match='line up one to one'):
            adapter.context_from_list(['only one'])
