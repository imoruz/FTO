from fto.adapters.node.node import (
    NodeAdapter,
    block_text,
    content_text,
    replace_content_text,
)


class TestNodeAdapterBase:
    def test_stores_the_inner_object(self):
        inner = object()
        adapter = NodeAdapter(inner)
        assert adapter._inner is inner

    def test_properties_are_stubs_returning_none(self):
        adapter = NodeAdapter(object())
        assert adapter.id is None
        assert adapter.input is None
        assert adapter.last_message is None
        assert adapter.is_agent is None

    def test_methods_are_noops(self):
        adapter = NodeAdapter(object())
        assert adapter.set_input('value') is None
        assert adapter.to_aegis_context() is None
        assert adapter.append_to_last_message('text') is None
        assert adapter.overwrite_last_message('text') is None


class TestNodeAdapterBaseContextStubs:
    def test_context_helpers_are_stubs(self):
        adapter = NodeAdapter(object())
        assert adapter.context_as_list() is None
        assert adapter.context_from_list(['text']) is None


class FakeBlock:
    def __init__(self, type_, text=None):
        self.type = type_
        self.text = text


class TestBlockText:
    def test_reads_the_text_of_an_object_block(self):
        assert block_text(FakeBlock('text', 'hello')) == 'hello'

    def test_reads_the_text_of_a_dict_block(self):
        assert block_text({'type': 'text', 'text': 'hello'}) == 'hello'

    def test_a_dict_block_is_text_by_default(self):
        assert block_text({'text': 'hello'}) == 'hello'

    def test_a_block_with_no_type_is_taken_as_text(self):
        block = type('Bare', (), {'text': 'hello'})()
        assert block_text(block) == 'hello'

    def test_non_text_blocks_have_no_text(self):
        assert block_text(FakeBlock('image', 'alt text')) == ''
        assert block_text({'type': 'image', 'text': 'alt text'}) == ''

    def test_missing_text_reads_as_empty(self):
        assert block_text(FakeBlock('text')) == ''
        assert block_text({'type': 'text'}) == ''
        assert block_text(object()) == ''


class TestContentText:
    def test_string_content_is_its_own_text(self):
        assert content_text('hello') == 'hello'

    def test_block_list_content_joins_its_text_blocks(self):
        content = [FakeBlock('text', 'a'), FakeBlock('image'), FakeBlock('text', 'b')]
        assert content_text(content) == 'a\n\nb'

    def test_content_of_an_unknown_shape_has_no_text(self):
        assert content_text(None) == ''
        assert content_text(12345) == ''

    def test_empty_content_has_no_text(self):
        assert content_text('') == ''
        assert content_text([]) == ''


def _dict_block(block, text):
    return {'type': 'text', 'text': text}


class TestReplaceContentText:
    def test_string_content_is_replaced_outright(self):
        assert replace_content_text('old', 'new', _dict_block) == 'new'

    def test_the_new_text_takes_the_first_text_slot(self):
        content = [FakeBlock('text', 'a'), FakeBlock('text', 'b')]
        assert replace_content_text(content, 'new', _dict_block) == [
            {'type': 'text', 'text': 'new'}
        ]

    def test_non_text_blocks_keep_their_place(self):
        image = FakeBlock('image')
        content = [image, FakeBlock('text', 'a')]

        rebuilt = replace_content_text(content, 'new', _dict_block)

        assert rebuilt[0] is image
        assert rebuilt[1] == {'type': 'text', 'text': 'new'}

    def test_content_with_no_text_to_replace_returns_none(self):
        assert replace_content_text([FakeBlock('image')], 'new', _dict_block) is None
        assert replace_content_text([], 'new', _dict_block) is None

    def test_content_of_an_unknown_shape_returns_none(self):
        assert replace_content_text(12345, 'new', _dict_block) is None
        assert replace_content_text(None, 'new', _dict_block) is None
