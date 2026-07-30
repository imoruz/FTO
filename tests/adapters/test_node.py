from fto.adapters.node.node import NodeAdapter


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
