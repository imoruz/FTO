import pytest

from fto.recovery.compression import CompressionResult, ContextCompressor
from fto.recovery.restart import (
    Restart,
    RestartAllContext,
    RestartNoContext,
    RestartRefinedContext,
)


class FakeMessage:
    """Message envelope whose text can be swapped without losing its role."""

    def __init__(self, role, text):
        self.role = role
        self.text = text

    def with_text(self, text):
        return FakeMessage(self.role, text)

    def __eq__(self, other):
        return (
            isinstance(other, FakeMessage)
            and (self.role, self.text) == (other.role, other.text)
        )

    def __repr__(self):
        return f'FakeMessage({self.role!r}, {self.text!r})'


class FakeAdapter:
    """Minimal stand-in for a NodeAdapter's context <-> text translation."""

    def __init__(self):
        self.as_list_calls = []
        self.from_list_calls = []

    def context_as_list(self, context=None):
        self.as_list_calls.append(context)
        return [message.text for message in context]

    def context_from_list(self, texts, context=None):
        self.from_list_calls.append((texts, context))
        return [
            message.with_text(text) if text else message
            for message, text in zip(context, texts)
        ]


class FakeCompressor(ContextCompressor):
    def __init__(self, texts=None):
        self.calls = []
        self._texts = texts

    def compress(self, contexts, question=''):
        self.calls.append((list(contexts), question))
        texts = self._texts if self._texts is not None else [f'<{c}>' for c in contexts]
        return CompressionResult(list(texts), origin_tokens=100, compressed_tokens=40)


class RecordingLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg, node_id=None):
        self.messages.append(msg)


def make_context(*pairs):
    return [FakeMessage(role, text) for role, text in pairs]


class TestRestartBase:
    def test_default_restart_count_is_one(self):
        assert Restart().restart_count == 1

    def test_restart_count_is_configurable(self):
        assert Restart(restart_count=5).restart_count == 5

    def test_set_context_deep_copies_the_context(self):
        restart = Restart()
        original = {'a': [1, 2, 3]}

        restart.set_context(original)

        original['a'].append(4)
        assert restart.context == {'a': [1, 2, 3]}
        assert restart.context is not original

    def test_set_context_keeps_the_adapter_by_reference(self):
        restart = Restart()
        adapter = FakeAdapter()

        restart.set_context([], adapter=adapter)

        assert restart.adapter is adapter

    def test_the_adapter_is_optional(self):
        restart = Restart()
        restart.set_context({'a': 1})
        assert restart.adapter is None

    def test_get_context_is_a_stub_returning_none(self):
        restart = Restart()
        restart.set_context({'a': 1})
        assert restart.get_context() is None


class TestRestartAllContext:
    def test_get_context_returns_the_stored_context(self):
        restart = RestartAllContext(restart_count=3)
        restart.set_context({'history': ['step1', 'step2']})

        assert restart.get_context() == {'history': ['step1', 'step2']}

    def test_get_context_returns_the_deep_copied_object(self):
        restart = RestartAllContext()
        original = {'a': [1]}
        restart.set_context(original)

        assert restart.get_context() is restart.context
        assert restart.get_context() is not original


class TestRestartNoContext:
    def test_get_context_always_returns_an_empty_context(self):
        restart = RestartNoContext()
        restart.set_context({'history': ['step1', 'step2']})

        assert restart.get_context() == []

    def test_get_context_returns_an_empty_context_without_set_context(self):
        assert RestartNoContext().get_context() == []


class TestRestartRefinedContext:
    def test_default_restart_count_is_one(self):
        assert RestartRefinedContext().restart_count == 1

    def test_defaults_to_an_llmlingua_compressor(self):
        from fto.recovery.compression import LLMLinguaCompressor

        assert isinstance(RestartRefinedContext().compressor, LLMLinguaCompressor)

    def test_keeps_at_least_the_last_message_verbatim(self):
        assert RestartRefinedContext(keep_last=0).keep_last == 1
        assert RestartRefinedContext(keep_last=-3).keep_last == 1

    def test_compresses_the_history_and_keeps_the_last_message(self):
        restart = RestartRefinedContext(compressor=FakeCompressor())
        restart.set_context(
            make_context(
                ('user', 'the issue'),
                ('assistant', 'the plan'),
                ('user', 'the report'),
            ),
            adapter=FakeAdapter(),
        )

        assert restart.get_context() == make_context(
            ('user', '<the issue>'),
            ('assistant', '<the plan>'),
            ('user', 'the report'),
        )

    def test_conditions_compression_on_the_message_kept_verbatim(self):
        compressor = FakeCompressor()
        restart = RestartRefinedContext(compressor=compressor)
        restart.set_context(
            make_context(('user', 'the plan'), ('user', 'the review')),
            adapter=FakeAdapter(),
        )

        restart.get_context()

        assert compressor.calls == [(['the plan'], 'the review')]

    def test_keep_last_can_hold_back_more_than_one_message(self):
        restart = RestartRefinedContext(compressor=FakeCompressor(), keep_last=2)
        restart.set_context(
            make_context(
                ('user', 'the issue'),
                ('assistant', 'the plan'),
                ('user', 'the report'),
                ('assistant', 'the review'),
            ),
            adapter=FakeAdapter(),
        )

        assert restart.get_context() == make_context(
            ('user', '<the issue>'),
            ('assistant', '<the plan>'),
            ('user', 'the report'),
            ('assistant', 'the review'),
        )

    def test_a_history_that_compresses_to_nothing_keeps_its_original_text(self):
        restart = RestartRefinedContext(compressor=FakeCompressor(texts=['']))
        restart.set_context(
            make_context(('user', 'the plan'), ('user', 'the review')),
            adapter=FakeAdapter(),
        )

        assert restart.get_context() == make_context(
            ('user', 'the plan'), ('user', 'the review')
        )

    def test_a_context_of_only_kept_messages_is_returned_as_is(self):
        compressor = FakeCompressor()
        restart = RestartRefinedContext(compressor=compressor)
        context = make_context(('user', 'the issue'))
        restart.set_context(context, adapter=FakeAdapter())

        assert restart.get_context() == context
        assert compressor.calls == []

    def test_an_empty_context_is_returned_as_is(self):
        restart = RestartRefinedContext(compressor=FakeCompressor())
        restart.set_context([], adapter=FakeAdapter())

        assert restart.get_context() == []

    def test_it_refines_the_snapshot_not_the_adapter_live_input(self):
        adapter = FakeAdapter()
        restart = RestartRefinedContext(compressor=FakeCompressor())
        context = make_context(('user', 'the plan'), ('user', 'the review'))
        restart.set_context(context, adapter=adapter)

        restart.get_context()

        # The snapshot taken before the fault is what gets read and rebuilt,
        # never whatever the fault left in the node's input.
        assert adapter.as_list_calls == [restart.context]
        assert adapter.from_list_calls[0][1] is restart.context

    def test_without_an_adapter_it_refuses_rather_than_silently_not_refining(self):
        restart = RestartRefinedContext(compressor=FakeCompressor())
        restart.set_context(make_context(('user', 'a'), ('user', 'b')))

        with pytest.raises(ValueError, match='needs the node adapter'):
            restart.get_context()

    def test_the_compression_result_is_kept_for_inspection(self):
        restart = RestartRefinedContext(compressor=FakeCompressor())
        restart.set_context(
            make_context(('user', 'the plan'), ('user', 'the review')),
            adapter=FakeAdapter(),
        )

        restart.get_context()

        assert restart.last_result.origin_tokens == 100
        assert restart.last_result.compressed_tokens == 40

    def test_it_logs_what_it_compressed(self):
        logger = RecordingLogger()
        restart = RestartRefinedContext(compressor=FakeCompressor(), logger=logger)
        restart.set_context(
            make_context(('user', 'a'), ('user', 'b'), ('user', 'c')),
            adapter=FakeAdapter(),
        )

        restart.get_context()

        assert logger.messages == [
            'Refined context: compressed 2 of 3 message(s), '
            '100 -> 40 tokens (40.0% of original); '
            'last 1 message(s) kept verbatim.'
        ]

    def test_it_works_without_a_logger(self):
        restart = RestartRefinedContext(compressor=FakeCompressor())
        restart.set_context(
            make_context(('user', 'a'), ('user', 'b')), adapter=FakeAdapter()
        )

        assert restart.get_context() is not None

    def test_it_compresses_once_and_replays_for_later_attempts(self):
        compressor = FakeCompressor()
        restart = RestartRefinedContext(restart_count=3, compressor=compressor)
        restart.set_context(
            make_context(('user', 'the plan'), ('user', 'the review')),
            adapter=FakeAdapter(),
        )

        first = restart.get_context()
        second = restart.get_context()

        assert len(compressor.calls) == 1
        assert first == second

    def test_every_attempt_gets_its_own_message_list(self):
        # A restarted node may mutate the input it was handed; the next attempt
        # must not inherit that.
        restart = RestartRefinedContext(restart_count=2, compressor=FakeCompressor())
        restart.set_context(
            make_context(('user', 'the plan'), ('user', 'the review')),
            adapter=FakeAdapter(),
        )

        first = restart.get_context()
        first.append(FakeMessage('user', 'something the node added'))

        assert len(restart.get_context()) == 2

    def test_a_new_snapshot_is_compressed_afresh(self):
        compressor = FakeCompressor()
        restart = RestartRefinedContext(compressor=compressor)
        adapter = FakeAdapter()

        restart.set_context(
            make_context(('user', 'first plan'), ('user', 'first review')),
            adapter=adapter,
        )
        restart.get_context()
        restart.set_context(
            make_context(('user', 'second plan'), ('user', 'second review')),
            adapter=adapter,
        )
        restart.get_context()

        assert compressor.calls == [
            (['first plan'], 'first review'),
            (['second plan'], 'second review'),
        ]
