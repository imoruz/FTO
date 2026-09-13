"""Structure-aware compression of a labelled agent message."""

from dataclasses import dataclass, field

import pytest

from fto.recovery.compression import (
    CompressionResult,
    ContextCompressor,
    SectionPolicy,
    StructuredCompressor,
    normalise_label,
)


@dataclass
class FakeTemplate(ContextCompressor):
    """Stands in for LLMLinguaCompressor: keeps the first `rate` of the words.

    `StructuredCompressor` clones the template per section with
    `dataclasses.replace`, so `calls` is a field rather than set in
    `__post_init__` -- that way every clone records into the same list.
    """

    rate: float = 1.0
    calls: list = field(default_factory=list)

    def compress(self, contexts, question=''):
        self.calls.append((self.rate, list(contexts)))
        texts = []
        for context in contexts:
            words = context.split()
            texts.append(' '.join(words[: max(1, round(len(words) * self.rate))]))
        return CompressionResult(texts)

    def _count(self, texts):
        return sum(len(t.split()) for t in texts)


# The labels this MAS's agents actually emit, with the plan's own fields held
# at a gentler rate than its narration.
SECTIONS = [
    SectionPolicy('THOUGHT', 0.3),
    SectionPolicy('ACTION', 0.9),
    SectionPolicy('OBSERVATION', 0.5),
    SectionPolicy('REPORT', 0.9),
    SectionPolicy('EDITS', 0.9),
    SectionPolicy('PLAN_DEVIATIONS', 0.9),
    SectionPolicy('CONCERNS', None),
    SectionPolicy('REMAINING', None),
    SectionPolicy('## Fix Plan', 0.9),
]

CODER_REPORT = (
    'THOUGHT: one two three four five six seven eight\n\n'
    'ACTION: alpha beta gamma delta epsilon zeta\n\n'
    'OBSERVATION: aa bb cc dd ee ff\n\n'
    'REPORT:\n'
    'EDITS:           changed the vendor branch and its helper\n'
    'PLAN_DEVIATIONS: none at all in this turn\n\n'
    'CONCERNS:        the fallback path must never be reached\n\n'
    'REMAINING: none\n\n'
    'READY_FOR_REVIEW'
)


def make(**kwargs):
    kwargs.setdefault('sections', SECTIONS)
    kwargs.setdefault('template', FakeTemplate())
    return StructuredCompressor(**kwargs)


class TestLabelNormalisation:
    def test_a_trailing_colon_and_spaces_are_ignored(self):
        assert normalise_label('EDITS:           ') == 'edits'
        assert normalise_label('EDITS') == 'edits'

    def test_a_heading_keeps_its_hashes(self):
        assert normalise_label('## Fix Plan') == '## fix plan'


class TestSectionSplitting:
    def test_every_label_starts_its_own_section(self):
        assert make().labels_of(CODER_REPORT) == [
            'THOUGHT: ',
            'ACTION: ',
            'OBSERVATION: ',
            'REPORT:',
            'EDITS:           ',
            'PLAN_DEVIATIONS: ',
            'CONCERNS:        ',
            'REMAINING: ',
        ]

    def test_each_section_carries_its_own_rate(self):
        rates = [rate for _, _, rate in make()._sections(CODER_REPORT)]
        assert rates == [0.3, 0.9, 0.5, 0.9, 0.9, 0.9, None, None]

    def test_a_preamble_before_the_first_label_is_its_own_piece(self):
        pieces = make(default_rate=0.5)._sections('loose words here\n\nTHOUGHT: a b')
        assert pieces[0] == ['', 'loose words here\n\n', 0.5]

    def test_a_label_must_be_followed_by_a_colon(self):
        # A line of prose that merely starts with the word is not a section.
        pieces = make()._sections('ACTION taken earlier was wrong\n\nACTION: a b c')
        assert [label for label, _, _ in pieces if label] == ['ACTION: ']

    def test_a_longer_label_wins_over_a_prefix_of_it(self):
        compressor = make(
            sections=[
                SectionPolicy('CONCERNS', 0.5),
                SectionPolicy('## CONCERNS Review', 0.9),
            ]
        )
        assert compressor.labels_of('## CONCERNS Review\nx y z') == [
            '## CONCERNS Review'
        ]

    def test_matching_ignores_case(self):
        assert make().labels_of('thought: a b c') == ['thought: ']

    def test_an_unlisted_subheading_stays_inside_its_section(self):
        text = 'ACTION: do it\n\n### Some Detail\nmore words here'
        bodies = [body for label, body, _ in make()._sections(text) if label]
        assert '### Some Detail' in bodies[0]

    def test_a_message_with_no_labels_is_one_piece(self):
        assert make()._sections('just prose') == [['', 'just prose', None]]

    def test_an_empty_message_has_no_pieces(self):
        assert make()._sections('') == []


class TestStructuredCompression:
    def test_the_labels_come_back_exactly_as_written(self):
        out = make().compress([CODER_REPORT]).texts[0]
        for label in (
            'THOUGHT:',
            'ACTION:',
            'OBSERVATION:',
            'REPORT:',
            'EDITS:',
            'PLAN_DEVIATIONS:',
            'CONCERNS:',
            'REMAINING:',
        ):
            assert label in out, label

    def test_the_sections_stay_in_order(self):
        out = make().compress([CODER_REPORT]).texts[0]
        order = [
            out.index(label)
            for label in ('THOUGHT:', 'ACTION:', 'OBSERVATION:', 'REPORT:', 'CONCERNS:')
        ]
        assert order == sorted(order)

    def test_a_section_with_no_rate_is_untouched(self):
        out = make().compress([CODER_REPORT]).texts[0]
        assert 'the fallback path must never be reached' in out
        assert 'REMAINING: none' in out

    def test_a_section_with_a_rate_is_compressed(self):
        out = make().compress([CODER_REPORT]).texts[0]
        # THOUGHT is 8 words at rate 0.3 -> 2 kept.
        assert 'THOUGHT: one two\n' in out

    def test_each_section_is_compressed_at_its_own_rate(self):
        out = make().compress([CODER_REPORT]).texts[0]
        # ACTION is 6 words at 0.9 -> 5; OBSERVATION 6 at 0.5 -> 3.
        assert 'ACTION: alpha beta gamma delta epsilon\n' in out
        assert 'OBSERVATION: aa bb cc\n' in out

    def test_the_terminal_marker_survives(self):
        out = make().compress([CODER_REPORT]).texts[0]
        assert 'READY_FOR_REVIEW' in out

    def test_the_message_gets_shorter(self):
        out = make().compress([CODER_REPORT]).texts[0]
        assert len(out) < len(CODER_REPORT)

    def test_one_model_call_per_distinct_rate_not_per_section(self):
        template = FakeTemplate()
        make(template=template).compress([CODER_REPORT])

        rates = [rate for rate, _ in template.calls]
        assert sorted(rates) == [0.3, 0.5, 0.9]

    def test_bodies_of_the_same_rate_are_batched_together(self):
        template = FakeTemplate()
        make(template=template).compress([CODER_REPORT])

        batched = next(bodies for rate, bodies in template.calls if rate == 0.9)
        # ACTION, EDITS, PLAN_DEVIATIONS. Not REPORT: its body is just the
        # newline before EDITS, and a body with no text is never sent.
        assert len(batched) == 3

    def test_several_entries_keep_their_alignment(self):
        result = make().compress([CODER_REPORT, 'THOUGHT: aa bb cc dd', ''])

        assert len(result.texts) == 3
        assert result.texts[1].startswith('THOUGHT: ')
        assert result.texts[2] == ''

    def test_unlabelled_text_is_kept_when_no_default_rate_is_set(self):
        assert make().compress(['plain prose with no labels at all']).texts == [
            'plain prose with no labels at all'
        ]

    def test_a_default_rate_compresses_unlabelled_text(self):
        out = make(default_rate=0.5).compress(['one two three four']).texts[0]
        assert out == 'one two'

    def test_no_sections_configured_falls_back_to_the_default_rate(self):
        compressor = StructuredCompressor(
            sections=[], default_rate=0.5, template=FakeTemplate()
        )
        assert compressor.compress(['one two three four']).texts == ['one two']

    def test_nothing_to_compress_returns_the_entries_untouched(self):
        template = FakeTemplate()
        result = StructuredCompressor(sections=SECTIONS, template=template).compress(
            ['CONCERNS: keep me', 'REMAINING: none']
        )

        assert result.texts == ['CONCERNS: keep me', 'REMAINING: none']
        assert template.calls == []

    def test_it_reports_token_counts_over_the_whole_entry(self):
        result = make().compress([CODER_REPORT])
        assert result.origin_tokens > result.compressed_tokens > 0

    def test_it_inherits_whether_the_template_uses_the_question(self):
        assert make().uses_question is False


class TestWhitespaceHandling:
    def test_a_body_is_not_welded_onto_its_label(self):
        out = make().compress(['THOUGHT: one two three four']).texts[0]
        assert out.startswith('THOUGHT: one')

    def test_blank_lines_between_sections_survive(self):
        out = make().compress([CODER_REPORT]).texts[0]
        assert '\n\n' in out

    def test_a_heading_keeps_its_own_line(self):
        out = make().compress(['## Fix Plan\nchange x then y then z now']).texts[0]
        assert out.startswith('## Fix Plan\n')


class TestRestartIntegration:
    """The restart layer decides whether the newest message is kept verbatim
    or compressed along its structure."""

    def _restart(self, **kwargs):
        from fto.recovery import RestartRefinedContext
        from tests.recovery.test_restart import (
            FakeAdapter,
            FakeCompressor,
            make_context,
        )

        restart = RestartRefinedContext(compressor=FakeCompressor(), **kwargs)
        restart.set_context(
            make_context(
                ('user', 'the task'),
                ('assistant', 'the plan'),
                ('user', CODER_REPORT),
            ),
            adapter=FakeAdapter(),
        )
        return restart

    def test_without_an_active_compressor_the_tail_is_verbatim(self):
        refined = self._restart().get_context()
        assert refined[2].text == CODER_REPORT

    def test_with_one_the_tail_is_compressed_along_its_structure(self):
        refined = self._restart(active_compressor=make()).get_context()

        assert refined[2].text != CODER_REPORT
        assert len(refined[2].text) < len(CODER_REPORT)
        for label in ('THOUGHT:', 'ACTION:', 'CONCERNS:', 'REMAINING:'):
            assert label in refined[2].text

    def test_the_history_is_still_compressed_the_ordinary_way(self):
        refined = self._restart(active_compressor=make()).get_context()
        assert refined[1].text == '<the plan>'

    def test_the_instruction_is_still_protected(self):
        refined = self._restart(active_compressor=make()).get_context()
        assert refined[0].text == 'the task'

    def test_the_tail_policy_is_recorded(self):
        restart = self._restart(active_compressor=make())
        restart.get_context()

        assert [r.policy for r in restart.records] == [
            'verbatim-head',
            'compressed',
            'compressed-structured',
        ]

    def test_the_log_says_the_tail_was_compressed(self):
        from tests.recovery.test_restart import RecordingLogger

        logger = RecordingLogger()
        restart = self._restart(active_compressor=make(), logger=logger)
        restart.get_context()

        assert 'compressed along its structure' in logger.messages[0]

    def test_the_tail_is_validated_like_any_other_message(self):
        # A structured compressor that eats a negation must be rejected too.
        class Eats(ContextCompressor):
            def compress(self, contexts, question=''):
                return CompressionResult(
                    [c.replace('must never', 'must') for c in contexts]
                )

        restart = self._restart(active_compressor=Eats())
        refined = restart.get_context()

        assert refined[2].text == CODER_REPORT
        assert [r.policy for r in restart.records][-1] == 'rejected'

    def test_the_reported_ratio_covers_both_zones(self):
        restart = self._restart(active_compressor=make())
        restart.get_context()

        # FakeCompressor reports 100 -> 40 for the history; the structured
        # compressor adds the tail's own counts on top.
        assert restart.last_result.origin_tokens > 100
        assert restart.last_result.compressed_tokens > 40


@pytest.mark.parametrize(
    'label',
    [
        'THOUGHT',
        'ACTION',
        'OBSERVATION',
        'OUTPUT',
        'REPORT',
        'EDITS',
        'PLAN_DEVIATIONS',
        'CONCERNS',
        'REMAINING',
    ],
)
def test_every_label_this_mas_emits_is_matchable(label):
    """Guards against a label the agents emit that the splitter cannot see."""
    compressor = StructuredCompressor(
        sections=[SectionPolicy(label, 0.5)], template=FakeTemplate()
    )
    assert compressor.labels_of(f'{label}: some body text here') == [f'{label}: ']


class TestFirstTurnRestart:
    """A node restarted on its first turn has exactly one message.

    That message is simultaneously the oldest and the newest. Giving the
    protected head priority left the tail empty, so a first-turn restart
    compressed nothing at all and behaved like RestartAllContext -- observed
    on a real run where the restarted Coder got back the Planner's 10,178
    character plan byte for byte.
    """

    def _restart(self, messages, **kwargs):
        from fto.recovery import RestartRefinedContext
        from tests.recovery.test_restart import (
            FakeAdapter,
            FakeCompressor,
            make_context,
        )

        restart = RestartRefinedContext(compressor=FakeCompressor(), **kwargs)
        restart.set_context(make_context(*messages), adapter=FakeAdapter())
        return restart

    def test_the_only_message_is_the_active_one_not_the_head(self):
        restart = self._restart([('user', CODER_REPORT)], active_compressor=make())

        refined = restart.get_context()

        assert refined[0].text != CODER_REPORT, 'the lone message was not compressed'
        assert [r.policy for r in restart.records] == ['compressed-structured']

    def test_its_structure_still_survives(self):
        restart = self._restart([('user', CODER_REPORT)], active_compressor=make())

        out = restart.get_context()[0].text

        for label in ('THOUGHT:', 'ACTION:', 'CONCERNS:', 'REMAINING:'):
            assert label in out, label
        assert 'the fallback path must never be reached' in out

    def test_the_turn_is_recorded_as_compressed(self):
        restart = self._restart([('user', CODER_REPORT)], active_compressor=make())

        restart.get_context()

        assert restart.compressed is True
        assert restart.failure is None

    def test_without_a_tail_compressor_there_is_still_nothing_to_do(self):
        # One message, kept verbatim: correct, and correctly reported.
        restart = self._restart([('user', CODER_REPORT)])

        assert restart.get_context()[0].text == CODER_REPORT
        assert restart.compressed is False
        assert 'nothing to compress' in restart.failure

    def test_two_messages_split_the_same_way_as_before(self):
        # Only the one-message case changes; the head still wins from two up.
        restart = self._restart(
            [('user', 'the task'), ('user', CODER_REPORT)], active_compressor=make()
        )

        restart.get_context()

        assert [r.policy for r in restart.records] == [
            'verbatim-head',
            'compressed-structured',
        ]

    def test_three_messages_still_put_the_plan_in_the_middle(self):
        restart = self._restart(
            [('user', 'the task'), ('assistant', 'the plan'), ('user', CODER_REPORT)],
            active_compressor=make(),
        )

        restart.get_context()

        assert [r.policy for r in restart.records] == [
            'verbatim-head',
            'compressed',
            'compressed-structured',
        ]

    def test_an_empty_history_does_not_call_the_history_compressor(self):
        from tests.recovery.test_restart import FakeCompressor

        history = FakeCompressor()
        restart = self._restart([('user', CODER_REPORT)], active_compressor=make())
        restart.compressor = history
        restart.set_context(restart.context, adapter=restart.adapter)

        restart.get_context()

        assert history.calls == []

    def test_a_failing_tail_compressor_obeys_the_passthrough_policy(self):
        class Boom(ContextCompressor):
            def compress(self, contexts, question=''):
                raise RuntimeError('boom')

        restart = self._restart(
            [('user', CODER_REPORT)], active_compressor=Boom(), on_error='passthrough'
        )

        assert restart.get_context()[0].text == CODER_REPORT
        assert restart.compressed is False
        assert 'boom' in restart.failure

    def test_a_failing_tail_compressor_raises_by_default(self):
        class Boom(ContextCompressor):
            def compress(self, contexts, question=''):
                raise RuntimeError('boom')

        restart = self._restart([('user', CODER_REPORT)], active_compressor=Boom())

        with pytest.raises(RuntimeError, match='boom'):
            restart.get_context()
