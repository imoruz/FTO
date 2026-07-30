from fto.adapters.node.chatdev_node import ChatDevNodeAdapter


class FakeBlock:
    """Stand-in for the real MessageBlock used inside list-content messages."""

    def __init__(self, text):
        self.text = text


class FakeMessage:
    def __init__(self, content):
        self.content = content

    def with_content(self, new_content):
        return FakeMessage(new_content)


class FakeInner:
    def __init__(self, id_, input_, type_='agent', role='a role', tools=None,
                 description='desc', model_name='gpt-4'):
        self.id = id_
        self.input = input_
        self.type = type_
        self.role = role
        self.tools = tools or [{'name': 'search'}]
        self.description = description
        self.model_name = model_name


def _stub_message_block(fake_module):
    """Register a fake entity.messages module whose MessageBlock is the same
    class as tests/adapters' FakeBlock, since chatdev_node.py imports MessageBlock
    lazily and does isinstance() / classmethod checks against it."""

    class MessageBlock(FakeBlock):
        @classmethod
        def text_block(cls, text):
            return cls(text)

    fake_module('entity.messages', MessageBlock=MessageBlock)
    return MessageBlock


class TestChatDevNodeAdapter:
    def test_id_returns_inner_id(self):
        adapter = ChatDevNodeAdapter(FakeInner('node-1', []))
        assert adapter.id == 'node-1'

    def test_input_returns_inner_input(self):
        messages = [FakeMessage('hi')]
        adapter = ChatDevNodeAdapter(FakeInner('node-1', messages))
        assert adapter.input is messages

    def test_set_input_replaces_inner_input(self):
        inner = FakeInner('node-1', [FakeMessage('old')])
        adapter = ChatDevNodeAdapter(inner)
        new_messages = [FakeMessage('new')]

        adapter.set_input(new_messages)

        assert inner.input is new_messages

    def test_is_agent_reflects_inner_type(self):
        assert ChatDevNodeAdapter(FakeInner('n', [], type_='agent')).is_agent is True
        assert ChatDevNodeAdapter(FakeInner('n', [], type_='tool')).is_agent is False

    def test_last_message_with_string_content(self):
        adapter = ChatDevNodeAdapter(FakeInner('n', [FakeMessage('hello')]))
        assert adapter.last_message == 'hello'

    def test_last_message_with_block_list_content(self):
        adapter = ChatDevNodeAdapter(FakeInner('n', [FakeMessage([FakeBlock('a'), FakeBlock('b')])]))
        assert adapter.last_message == 'b'

    def test_last_message_is_none_when_no_input(self):
        adapter = ChatDevNodeAdapter(FakeInner('n', []))
        assert adapter.last_message is None

    def test_to_aegis_context_maps_inner_fields(self):
        inner = FakeInner('node-1', [FakeMessage('hi'), FakeMessage('there')],
                           type_='agent', role='sys msg', tools=[{'name': 'a'}, {'name': 'b'}],
                           description='does stuff', model_name='claude')
        adapter = ChatDevNodeAdapter(inner)

        ctx = adapter.to_aegis_context()

        assert ctx.role_type == 'agent'
        assert ctx.agent_id == 'node-1'
        assert ctx.system_message == 'sys msg'
        assert ctx.tools == ['a', 'b']
        assert ctx.description == 'does stuff'
        assert ctx.model_type == 'claude'
        assert ctx.recent_history == ['hi', 'there']

    def test_append_to_last_message_with_string_content(self, fake_module):
        _stub_message_block(fake_module)
        inner = FakeInner('n', [FakeMessage('hello')])
        adapter = ChatDevNodeAdapter(inner)

        adapter.append_to_last_message('world')

        assert inner.input[-1].content == 'hello\n\nworld'

    def test_append_to_last_message_with_block_list_content(self, fake_module):
        MessageBlock = _stub_message_block(fake_module)
        original_block = MessageBlock('first')
        inner = FakeInner('n', [FakeMessage([original_block])])
        adapter = ChatDevNodeAdapter(inner)

        adapter.append_to_last_message('second')

        content = inner.input[-1].content
        assert content[0] is original_block
        assert isinstance(content[1], MessageBlock)
        assert content[1].text == 'second'

    def test_append_to_last_message_with_dict_list_content(self, fake_module):
        _stub_message_block(fake_module)
        inner = FakeInner('n', [FakeMessage([{'type': 'text', 'text': 'first'}])])
        adapter = ChatDevNodeAdapter(inner)

        adapter.append_to_last_message('second')

        content = inner.input[-1].content
        assert content == [{'type': 'text', 'text': 'first'}, {'type': 'text', 'text': 'second'}]

    def test_append_to_last_message_noop_when_no_input(self, fake_module):
        _stub_message_block(fake_module)
        inner = FakeInner('n', [])
        adapter = ChatDevNodeAdapter(inner)

        adapter.append_to_last_message('anything')

        assert inner.input == []

    def test_append_to_last_message_noop_for_unknown_content_type(self, fake_module):
        _stub_message_block(fake_module)
        inner = FakeInner('n', [FakeMessage(12345)])
        adapter = ChatDevNodeAdapter(inner)

        adapter.append_to_last_message('anything')

        assert inner.input[-1].content == 12345

    def test_overwrite_last_message_with_string_content(self, fake_module):
        _stub_message_block(fake_module)
        inner = FakeInner('n', [FakeMessage('hello')])
        adapter = ChatDevNodeAdapter(inner)

        adapter.overwrite_last_message('replaced')

        assert len(inner.input) == 1
        assert inner.input[-1].content == 'replaced'

    def test_overwrite_last_message_with_block_list_content(self, fake_module):
        MessageBlock = _stub_message_block(fake_module)
        inner = FakeInner('n', [FakeMessage([MessageBlock('old')])])
        adapter = ChatDevNodeAdapter(inner)

        adapter.overwrite_last_message('replaced')

        content = inner.input[-1].content
        assert len(content) == 1
        assert isinstance(content[0], MessageBlock)
        assert content[0].text == 'replaced'

    def test_overwrite_last_message_noop_when_no_input(self, fake_module):
        _stub_message_block(fake_module)
        inner = FakeInner('n', [])
        adapter = ChatDevNodeAdapter(inner)

        adapter.overwrite_last_message('replaced')

        assert inner.input == []
