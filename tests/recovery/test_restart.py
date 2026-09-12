import pytest

from fto.recovery.compression import (
    CompressionResult,
    CompressionValidator,
    ContextCompressor,
)
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
    def __init__(self, texts=None, uses_question=False, raises=None):
        self.calls = []
        self._texts = texts
        self.uses_question = uses_question
        self._raises = raises


    def compress(self, contexts, question=''):
        self.calls.append((list(contexts), question))
        if self._raises is not None:
            raise self._raises
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
            ('user', 'the issue'),          # protected head (the task statement)
            ('assistant', '<the plan>'),    # compressed
            ('user', 'the report'),         # kept verbatim
        )

    def test_a_query_aware_compressor_is_given_the_message_kept_verbatim(self):
        compressor = FakeCompressor(uses_question=True)
        restart = RestartRefinedContext(compressor=compressor)
        restart.set_context(
            make_context(
                ('user', 'the task'), ('user', 'the plan'), ('user', 'the review')
            ),
            adapter=FakeAdapter(),
        )

        restart.get_context()

        assert compressor.calls == [(['the plan'], 'the review')]

    def test_no_question_is_built_for_a_compressor_that_ignores_it(self):
        # LLMLingua-2 is task-agnostic and discards the question; handing it
        # one would imply a conditioning that does not happen.
        compressor = FakeCompressor(uses_question=False)
        restart = RestartRefinedContext(compressor=compressor)
        restart.set_context(
            make_context(
                ('user', 'the task'), ('user', 'the plan'), ('user', 'the review')
            ),
            adapter=FakeAdapter(),
        )

        restart.get_context()

        assert compressor.calls == [(['the plan'], '')]

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
            ('user', 'the issue'),          # protected head
            ('assistant', '<the plan>'),    # compressed
            ('user', 'the report'),         # kept verbatim (keep_last=2)
            ('assistant', 'the review'),
        )

    def test_a_history_that_compresses_to_nothing_keeps_its_original_text(self):
        restart = RestartRefinedContext(compressor=FakeCompressor(texts=['']))
        restart.set_context(
            make_context(
                ('user', 'the task'), ('user', 'the plan'), ('user', 'the review')
            ),
            adapter=FakeAdapter(),
        )

        assert restart.get_context() == make_context(
            ('user', 'the task'), ('user', 'the plan'), ('user', 'the review')
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
        context = make_context(
            ('user', 'the task'), ('user', 'the plan'), ('user', 'the review')
        )
        restart.set_context(context, adapter=adapter)

        restart.get_context()

        # The snapshot taken before the fault is what gets read and rebuilt,
        # never whatever the fault left in the node's input.
        assert adapter.as_list_calls == [restart.context]
        assert adapter.from_list_calls[0][1] is restart.context

    def test_without_an_adapter_it_refuses_rather_than_silently_not_refining(self):
        restart = RestartRefinedContext(compressor=FakeCompressor())
        restart.set_context(
            make_context(('user', 'the task'), ('user', 'a'), ('user', 'b'))
        )

        with pytest.raises(ValueError, match='needs the node adapter'):
            restart.get_context()

    def test_the_compression_result_is_kept_for_inspection(self):
        restart = RestartRefinedContext(compressor=FakeCompressor())
        restart.set_context(
            make_context(
                ('user', 'the task'), ('user', 'the plan'), ('user', 'the review')
            ),
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
            'Refined context: compressed 1 of 3 message(s), '
            '100 -> 40 tokens (40.0% of original); '
            'first 1 kept as instruction, last 1 kept verbatim; '
            'question-conditioned: False.'
        ]

    def test_the_log_counts_only_messages_that_held_text(self):
        # Empty messages are never sent to the compressor; counting them as
        # compressed is what made "compressed 6 of 7" misleading in a run
        # where only two messages had any text.
        logger = RecordingLogger()
        restart = RestartRefinedContext(compressor=FakeCompressor(), logger=logger)
        restart.set_context(
            make_context(('user', 'a'), ('assistant', ''), ('assistant', ''),
                         ('user', 'b'), ('user', 'c')),
            adapter=FakeAdapter(),
        )

        restart.get_context()

        assert logger.messages[0].startswith('Refined context: compressed 1 of 5 message(s)')

    def test_it_works_without_a_logger(self):
        restart = RestartRefinedContext(compressor=FakeCompressor())
        restart.set_context(
            make_context(('user', 'the task'), ('user', 'a'), ('user', 'b')),
            adapter=FakeAdapter(),
        )

        assert restart.get_context() is not None

    def test_it_compresses_once_and_replays_for_later_attempts(self):
        compressor = FakeCompressor()
        restart = RestartRefinedContext(restart_count=3, compressor=compressor)
        restart.set_context(
            make_context(
                ('user', 'the task'), ('user', 'the plan'), ('user', 'the review')
            ),
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
            make_context(
                ('user', 'the task'), ('user', 'the plan'), ('user', 'the review')
            ),
            adapter=FakeAdapter(),
        )

        first = restart.get_context()
        first.append(FakeMessage('user', 'something the node added'))

        assert len(restart.get_context()) == 3

    def test_a_new_snapshot_is_compressed_afresh(self):
        compressor = FakeCompressor()
        restart = RestartRefinedContext(compressor=compressor)
        adapter = FakeAdapter()

        restart.set_context(
            make_context(('user', 'task'), ('user', 'first plan'),
                         ('user', 'first review')),
            adapter=adapter,
        )
        restart.get_context()
        restart.set_context(
            make_context(('user', 'task'), ('user', 'second plan'),
                         ('user', 'second review')),
            adapter=adapter,
        )
        restart.get_context()

        assert compressor.calls == [
            (['first plan'], ''),
            (['second plan'], ''),
        ]


class TestRestartRefinedContextFailurePolicy:
    """A benchmark must never contain an invisible mix of refined and not."""

    def _restart(self, **kwargs):
        restart = RestartRefinedContext(**kwargs)
        restart.set_context(
            make_context(
                ('user', 'the task'), ('user', 'the plan'), ('user', 'the review')
            ),
            adapter=FakeAdapter(),
        )
        return restart

    def test_it_raises_by_default(self):
        restart = self._restart(compressor=FakeCompressor(raises=RuntimeError('boom')))

        with pytest.raises(RuntimeError, match='boom'):
            restart.get_context()

    def test_passthrough_returns_the_uncompressed_history(self):
        restart = self._restart(
            compressor=FakeCompressor(raises=RuntimeError('boom')),
            on_error='passthrough',
        )

        assert restart.get_context() == make_context(
            ('user', 'the task'), ('user', 'the plan'), ('user', 'the review')
        )

    def test_passthrough_marks_the_turn_uncompressed_and_says_why(self):
        restart = self._restart(
            compressor=FakeCompressor(raises=RuntimeError('boom')),
            on_error='passthrough',
        )

        restart.get_context()

        assert restart.compressed is False
        assert restart.failure == 'RuntimeError: boom'

    def test_passthrough_logs_a_warning_naming_the_consequence(self):
        logger = RecordingLogger()
        restart = self._restart(
            compressor=FakeCompressor(raises=RuntimeError('boom')),
            on_error='passthrough',
            logger=logger,
        )

        restart.get_context()

        assert 'NOT refined' in logger.messages[0]
        assert 'boom' in logger.messages[0]

    def test_a_raise_still_records_the_failure_before_propagating(self):
        restart = self._restart(compressor=FakeCompressor(raises=ValueError('nope')))

        with pytest.raises(ValueError):
            restart.get_context()

        assert restart.compressed is False
        assert restart.failure == 'ValueError: nope'

    def test_a_successful_compression_marks_the_turn_compressed(self):
        restart = self._restart(compressor=FakeCompressor())

        restart.get_context()

        assert restart.compressed is True
        assert restart.failure is None

    def test_metadata_starts_unset_and_resets_with_each_snapshot(self):
        restart = self._restart(compressor=FakeCompressor())
        restart.get_context()

        restart.set_context(make_context(('user', 'a')), adapter=FakeAdapter())

        assert restart.compressed is None
        assert restart.failure is None

    def test_a_history_of_only_kept_messages_is_recorded_as_uncompressed(self):
        restart = RestartRefinedContext(compressor=FakeCompressor())
        restart.set_context(make_context(('user', 'only this')), adapter=FakeAdapter())

        restart.get_context()

        assert restart.compressed is False
        assert 'nothing to compress' in restart.failure

    def test_an_unknown_policy_is_rejected_at_construction(self):
        with pytest.raises(ValueError):
            RestartRefinedContext(on_error='carry-on-regardless')


class TestRestartRefinedContextProtectsTheInstruction:
    """The oldest messages are instruction-like, and token pruning damages
    instructions worst. Measured on a real run: the task specification was the
    *most* aggressively compressed message in the context (to 62%), losing
    "DON'T have to modify the testing logic" and collapsing "Current Behavior"
    and "Expected Behavior" into the same label."""

    def _context(self):
        return make_context(
            ('user', 'the task specification'),
            ('assistant', 'the plan'),
            ('user', 'the report'),
            ('user', 'the review'),
        )

    def test_the_oldest_message_is_never_compressed(self):
        restart = RestartRefinedContext(compressor=FakeCompressor())
        restart.set_context(self._context(), adapter=FakeAdapter())

        refined = restart.get_context()

        assert refined[0].text == 'the task specification'

    def test_the_instruction_never_reaches_the_compressor(self):
        compressor = FakeCompressor()
        restart = RestartRefinedContext(compressor=compressor)
        restart.set_context(self._context(), adapter=FakeAdapter())

        restart.get_context()

        assert compressor.calls == [(['the plan', 'the report'], '')]

    def test_keep_first_can_protect_more_than_one_message(self):
        compressor = FakeCompressor()
        restart = RestartRefinedContext(compressor=compressor, keep_first=2)
        restart.set_context(self._context(), adapter=FakeAdapter())

        restart.get_context()

        assert compressor.calls == [(['the report'], '')]

    def test_keep_first_zero_compresses_the_instruction_too(self):
        # The old behaviour, kept available for an ablation.
        compressor = FakeCompressor()
        restart = RestartRefinedContext(compressor=compressor, keep_first=0)
        restart.set_context(self._context(), adapter=FakeAdapter())

        restart.get_context()

        assert compressor.calls[0][0][0] == 'the task specification'

    def test_head_and_tail_cannot_overlap(self):
        compressor = FakeCompressor()
        restart = RestartRefinedContext(compressor=compressor, keep_first=5)
        restart.set_context(self._context(), adapter=FakeAdapter())

        assert restart.get_context() == self._context()
        assert compressor.calls == []

    def test_a_context_with_no_middle_is_recorded_as_uncompressed(self):
        restart = RestartRefinedContext(compressor=FakeCompressor())
        restart.set_context(
            make_context(('user', 'the task'), ('user', 'the review')),
            adapter=FakeAdapter(),
        )

        restart.get_context()

        assert restart.compressed is False
        assert 'protected head' in restart.failure


class TestRestartRefinedContextValidator:
    """Token pruning cannot tell a faithful shortening from an inverted one, so
    a compressed message is checked before it is accepted."""

    def _restart(self, texts, **kwargs):
        restart = RestartRefinedContext(
            compressor=FakeCompressor(texts=texts), **kwargs
        )
        restart.set_context(
            make_context(
                ('user', 'the task'),
                ('assistant', "you must not skip `helpers.go` and DON'T guess"),
                ('user', 'the review'),
            ),
            adapter=FakeAdapter(),
        )
        return restart

    def test_a_lost_negation_is_rejected_and_the_original_kept(self):
        restart = self._restart(['you skip `helpers.go` and guess'])

        refined = restart.get_context()

        assert refined[1].text == "you must not skip `helpers.go` and DON'T guess"

    def test_a_rejection_is_recorded_with_its_reason(self):
        restart = self._restart(['you skip `helpers.go` and guess'])

        restart.get_context()

        rejected = [r for r in restart.records if r.policy == 'rejected']
        assert len(rejected) == 1
        assert any('negation' in f for f in rejected[0].failures)

    def test_a_lost_code_span_is_rejected(self):
        restart = self._restart(["you must not skip `helpers.` and DON'T guess"])

        restart.get_context()

        rejected = [r for r in restart.records if r.policy == 'rejected']
        assert any('code span' in f for f in rejected[0].failures)

    def test_a_faithful_shortening_is_accepted(self):
        restart = self._restart(["must not skip `helpers.go` DON'T guess"])

        refined = restart.get_context()

        assert refined[1].text == "must not skip `helpers.go` DON'T guess"
        assert [r.policy for r in restart.records if r.policy == 'rejected'] == []

    def test_a_rejection_is_logged(self):
        logger = RecordingLogger()
        restart = self._restart(['you skip `helpers.go` and guess'], logger=logger)

        restart.get_context()

        assert any('rejected a compressed message' in m for m in logger.messages)

    def test_the_validator_can_be_relaxed(self):
        restart = self._restart(
            ['you skip `helpers.go` and guess'],
            validator=CompressionValidator(negations=False, code_spans=False),
        )

        assert restart.get_context()[1].text == 'you skip `helpers.go` and guess'


class TestRestartRefinedContextBenefitGates:
    """Achieved ratio on a real run was 88.8% -- 523 tokens saved for the risk
    of corrupting an instruction. Both gates are off by default so an existing
    config keeps its behaviour."""

    def _restart(self, **kwargs):
        restart = RestartRefinedContext(compressor=FakeCompressor(), **kwargs)
        restart.set_context(
            make_context(
                ('user', 'the task'), ('assistant', 'the plan'), ('user', 'the review')
            ),
            adapter=FakeAdapter(),
        )
        return restart

    def test_both_gates_are_off_by_default(self):
        restart = self._restart()
        assert restart.min_chars == 0
        assert restart.min_saving == 0.0
        assert restart.compressed is None

    def test_a_history_below_min_chars_is_not_compressed(self):
        restart = self._restart(min_chars=10_000)

        refined = restart.get_context()

        assert refined[1].text == 'the plan'
        assert restart.compressed is False
        assert 'min_chars' in restart.failure

    def test_a_saving_below_min_saving_is_discarded(self):
        # FakeCompressor reports 100 -> 40 tokens, a 60% saving.
        restart = self._restart(min_saving=0.8)

        refined = restart.get_context()

        assert refined[1].text == 'the plan'
        assert restart.compressed is False
        assert 'min_saving' in restart.failure

    def test_a_saving_above_min_saving_is_kept(self):
        restart = self._restart(min_saving=0.5)

        assert restart.get_context()[1].text == '<the plan>'
        assert restart.compressed is True

    def test_a_skip_is_logged_and_recorded(self):
        logger = RecordingLogger()
        restart = self._restart(min_chars=10_000, logger=logger)

        restart.get_context()

        assert any('skipped compression' in m for m in logger.messages)
        assert [r.policy for r in restart.records] == [
            'verbatim-head', 'unchanged', 'verbatim-tail'
        ]


class TestRestartRefinedContextAuditRecords:
    def test_one_record_per_message_with_its_policy(self):
        restart = RestartRefinedContext(compressor=FakeCompressor())
        restart.set_context(
            make_context(
                ('user', 'the task'),
                ('assistant', 'the plan'),
                ('user', 'the report'),
                ('user', 'the review'),
            ),
            adapter=FakeAdapter(),
        )

        restart.get_context()

        assert [(r.index, r.policy) for r in restart.records] == [
            (0, 'verbatim-head'),
            (1, 'compressed'),
            (2, 'compressed'),
            (3, 'verbatim-tail'),
        ]

    def test_a_protected_message_is_never_marked_mutated(self):
        # A verbatim policy with mutated=True would be a bug, not a nuance.
        restart = RestartRefinedContext(compressor=FakeCompressor())
        restart.set_context(
            make_context(
                ('user', 'the task'), ('assistant', 'the plan'), ('user', 'the review')
            ),
            adapter=FakeAdapter(),
        )

        restart.get_context()

        for record in restart.records:
            if record.policy.startswith('verbatim'):
                assert not record.mutated, record

    def test_records_reset_with_each_snapshot(self):
        restart = RestartRefinedContext(compressor=FakeCompressor())
        adapter = FakeAdapter()
        restart.set_context(
            make_context(('user', 'a'), ('user', 'b'), ('user', 'c')), adapter=adapter
        )
        restart.get_context()

        restart.set_context(make_context(('user', 'x')), adapter=adapter)

        assert restart.records == []
