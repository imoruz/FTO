"""The pinned state block a restarted node is rebuilt on top of."""

import pytest

from fto.recovery.compression import SectionPolicy
from fto.recovery.resumption import (
    PinnedField,
    ResumptionPolicy,
    ResumptionState,
)

SECTIONS = [
    SectionPolicy('THOUGHT', 0.25),
    SectionPolicy('ACTION', 0.3),
    SectionPolicy('OBSERVATION', 0.45),
    SectionPolicy('## Fix Plan', 0.55),
    SectionPolicy('## Assumptions & Open Questions', None, pin='OPEN ASSUMPTIONS'),
    SectionPolicy('EDITS', 0.55, pin='EDIT LEDGER'),
    SectionPolicy('PLAN_DEVIATIONS', None, pin='PLAN_DEVIATIONS'),
    SectionPolicy('CONCERNS', None, pin='OPEN CONCERNS'),
    SectionPolicy('REMAINING', None, pin='REMAINING'),
]

PLAN = (
    'THOUGHT: This is my first turn so I am planning from scratch.\n\n'
    '## Fix Plan\n'
    '### File path: lib/ansible/utils/unsafe_proxy.py\n'
    '#### Change 1: widen the wrapper set\n'
    'Dispatch on the concrete type.\n\n'
    '## Assumptions & Open Questions\n'
    '- wrap_var is only called from task_executor.py\n'
    '- The helper is never handed a null argument\n'
)

REPORT = (
    'THOUGHT: The plan asked me to widen the wrapper set.\n\n'
    'ACTION: Applied the edits.\n\n'
    'OBSERVATION: Both hunks landed.\n\n'
    'REPORT:\n'
    'EDITS:           `lib/ansible/utils/unsafe_proxy.py` at lines 105-113, '
    'and the export at line 61.\n'
    'PLAN_DEVIATIONS: I also touched the import block.\n\n'
    'CONCERNS:        the fallback at line 233 must never be reached.\n\n'
    'REMAINING: none\n\n'
    'READY_FOR_REVIEW'
)

CONTEXT = ['the original issue text', PLAN, REPORT]


def policy(**kwargs):
    kwargs.setdefault('turn_marker_label', 'EDITS')
    kwargs.setdefault('loop_guard_limit', 8)
    kwargs.setdefault('trailing_markers', ['READY_FOR_REVIEW', 'TASK_COMPLETE'])
    return ResumptionPolicy.from_sections(SECTIONS, **kwargs)


def block(texts=CONTEXT, **kwargs):
    return ResumptionState(policy(**kwargs)).build(texts)


class TestPolicyFromSections:
    def test_only_pinned_sections_become_fields(self):
        titles = [f.title for f in policy().fields]
        assert titles == [
            'OPEN ASSUMPTIONS',
            'EDIT LEDGER',
            'PLAN_DEVIATIONS',
            'OPEN CONCERNS',
            'REMAINING',
        ]

    def test_the_edit_ledger_is_rendered_structurally_and_accumulates(self):
        edits = next(f for f in policy().fields if f.title == 'EDIT LEDGER')
        assert edits.ledger is True
        # Every turn's edits matter; the latest CONCERNS supersedes the rest.
        assert edits.latest_only is False

    def test_unpinned_labels_are_still_carried_as_boundaries(self):
        # A pinned section ends where the next section starts, pinned or not.
        assert 'REPORT' not in [f.label for f in policy().fields]


class TestResumptionMarker:
    def test_it_says_which_turn_this_is(self):
        assert 'This is turn 3, not your first turn' in block()

    def test_a_single_message_context_is_marked_as_a_discarded_attempt(self):
        # A node restarted on its very first turn genuinely is on turn one.
        out = block([PLAN])
        assert 'not your first turn' not in out
        assert 'discarded' in out

    def test_the_loop_guard_line_counts_the_marked_turns(self):
        assert 'LoopGuard: 1/8' in block()

    def test_it_says_a_restart_does_not_consume_a_turn(self):
        # Verified in the harness: the faulty attempt's outgoing edges are
        # withheld and only the successful attempt is released, so the
        # Coder -> LoopGuard edge fires once per logical turn.
        assert 'does not consume another' in block()

    def test_the_loop_guard_line_is_omitted_when_the_limit_is_unknown(self):
        assert 'LoopGuard' not in block(loop_guard_limit=None)


class TestPinnedFields:
    def test_assumptions_come_back_verbatim(self):
        out = block()
        assert '- wrap_var is only called from task_executor.py' in out
        assert '- The helper is never handed a null argument' in out

    def test_concerns_come_back_verbatim_with_their_turn(self):
        assert 'OPEN CONCERNS (turn 3):' in block()
        assert 'the fallback at line 233 must never be reached.' in block()

    def test_a_pinned_section_stops_at_the_next_label(self):
        # Without the unpinned labels as boundaries, CONCERNS swallows
        # REMAINING and everything after it.
        concerns = block().split('OPEN CONCERNS (turn 3):')[1]
        assert concerns.split('\n')[1].strip() == (
            'the fallback at line 233 must never be reached.'
        )

    def test_a_trailing_routing_marker_is_not_pinned(self):
        # READY_FOR_REVIEW trails REMAINING, the Coder's last field. It
        # belongs to the message, not to the field, and quoting a routing
        # keyword inside the state block is noise at best.
        assert 'READY_FOR_REVIEW' not in block()

    def test_deviations_come_back_verbatim(self):
        assert 'I also touched the import block.' in block()

    def test_the_block_never_ends_with_a_routing_keyword(self):
        assert block().rstrip().endswith('trust this block.')

    def test_the_authority_note_is_present(self):
        assert 'authoritative' in block()


class TestEditLedger:
    def test_it_renders_file_then_line_ranges(self):
        assert (
            '  lib/ansible/utils/unsafe_proxy.py -> lines 105-113, line 61' in block()
        )

    def test_it_names_the_turn_the_edit_came_from(self):
        assert '[turn 3]' in block()

    def test_a_quoted_path_is_read_like_a_bare_one(self):
        # The report writes paths in backticks; the ledger is about the path.
        assert '`' not in block().split('EDIT LEDGER:')[1].split('\n')[1]

    def test_a_file_with_no_line_reference_is_still_listed(self):
        out = block(['EDITS: rewrote helpers/format.py entirely'])
        assert 'helpers/format.py -> lines not stated' in out

    def test_edits_from_every_turn_accumulate(self):
        out = block(
            [
                'EDITS: `a/one.py` at line 5',
                'EDITS: `b/two.py` at line 9',
            ]
        )
        assert 'a/one.py -> line 5   [turn 1]' in out
        assert 'b/two.py -> line 9   [turn 2]' in out


class TestEmptyCases:
    def test_no_pinned_fields_means_no_block(self):
        empty = ResumptionPolicy(fields=[])
        assert ResumptionState(empty).build(CONTEXT) == ''

    def test_a_context_with_none_of_the_fields_means_no_block(self):
        assert block(['just some prose', 'and some more']) == ''

    def test_an_empty_context_means_no_block(self):
        assert block([]) == ''
        assert block(['', '']) == ''

    def test_an_empty_section_body_is_skipped(self):
        assert 'OPEN CONCERNS' not in block(['CONCERNS:\n\nREMAINING: none'])


class TestNoModelIsInvolved:
    def test_the_block_is_built_from_the_originals_only(self):
        """Z1 is regenerated structurally; nothing here may call a model."""
        state = ResumptionState(policy())
        # Every pinned body appears in the output character for character.
        out = state.build(CONTEXT)
        for needle in (
            '- wrap_var is only called from task_executor.py',
            'the fallback at line 233 must never be reached.',
            'I also touched the import block.',
        ):
            assert needle in out


@pytest.mark.parametrize(
    'label,title',
    [
        ('## Assumptions & Open Questions', 'OPEN ASSUMPTIONS'),
        ('CONCERNS', 'OPEN CONCERNS'),
        ('PLAN_DEVIATIONS', 'PLAN_DEVIATIONS'),
        ('REMAINING', 'REMAINING'),
    ],
)
def test_every_pinned_label_this_mas_emits_is_harvestable(label, title):
    state = ResumptionState(ResumptionPolicy(fields=[PinnedField(label, title)]))
    # A heading label runs to the end of its line; a field label takes a colon.
    written = (
        f'{label}\nthe body of it'
        if label.startswith('#')
        else (f'{label}: the body of it')
    )
    assert title in state.build([written])


# Sections that are pinned *and* verbatim are the ones that would otherwise
# appear twice. EDITS keeps its body: the block renders it as a summary
# ledger, not as the original prose.
POINTER = '(see {title} in the resumption state block above)'
POINTED_SECTIONS = [
    SectionPolicy('THOUGHT', 0.25),
    SectionPolicy('ACTION', 0.3),
    SectionPolicy('OBSERVATION', 0.45),
    SectionPolicy('## Fix Plan', 0.55),
    SectionPolicy(
        '## Assumptions & Open Questions',
        None,
        pin='OPEN ASSUMPTIONS',
        pin_pointer=POINTER,
    ),
    SectionPolicy('EDITS', 0.55, pin='EDIT LEDGER'),
    SectionPolicy('PLAN_DEVIATIONS', None, pin='PLAN_DEVIATIONS', pin_pointer=POINTER),
    SectionPolicy('CONCERNS', None, pin='OPEN CONCERNS', pin_pointer=POINTER),
    SectionPolicy('REMAINING', None, pin='REMAINING', pin_pointer=POINTER),
]


# A CONCERNS the length a real one is, so a pointer is genuinely shorter than
# the body it replaces. The shared REPORT's one-liner is not, by design --
# see test_a_body_shorter_than_the_pointer_is_left_alone.
LONG_CONCERNS = (
    'the fallback at line 233 must never be reached for a switch, because it '
    'assumes the router identifier and there is no guard on that branch.'
)
LONG_REPORT = REPORT.replace(
    'the fallback at line 233 must never be reached.', LONG_CONCERNS
)


def pointed_state(**kwargs):
    kwargs.setdefault('turn_marker_label', 'EDITS')
    kwargs.setdefault('loop_guard_limit', 8)
    kwargs.setdefault('trailing_markers', ['READY_FOR_REVIEW', 'TASK_COMPLETE'])
    return ResumptionState(ResumptionPolicy.from_sections(POINTED_SECTIONS, **kwargs))


class TestPinPointer:
    """A field pinned verbatim into the block should not also sit in the body.

    Measured on a real restart: the seven assumption items appeared twice,
    once in the block and once in the untouched tail, and the reconstructed
    prompt came out ~10% *larger* than the message it replaced.
    """

    def test_a_pointed_body_is_replaced_and_the_label_stays(self):
        state = pointed_state()
        state.build(['issue', PLAN, LONG_REPORT])

        out = state.strip_pinned(2, LONG_REPORT)

        assert 'CONCERNS:' in out
        assert LONG_CONCERNS not in out
        assert '(see OPEN CONCERNS in the resumption state block above)' in out

    def test_a_heading_label_keeps_its_line_break(self):
        state = pointed_state()
        state.build(CONTEXT)

        out = state.strip_pinned(1, PLAN)

        assert '## Assumptions & Open Questions\n(see OPEN ASSUMPTIONS' in out

    def test_a_field_label_stays_inline(self):
        state = pointed_state()
        state.build(['issue', PLAN, LONG_REPORT])

        out = state.strip_pinned(2, LONG_REPORT)

        assert 'CONCERNS:        (see OPEN CONCERNS' in out

    def test_a_pinned_section_without_a_pointer_keeps_its_body(self):
        # EDITS is pinned as a ledger summary, so the prose must stay.
        state = pointed_state()
        state.build(CONTEXT)

        out = state.strip_pinned(2, REPORT)

        assert 'and the export at line 61' in out

    def test_unpinned_sections_are_untouched(self):
        state = pointed_state()
        state.build(CONTEXT)

        out = state.strip_pinned(2, REPORT)

        assert 'ACTION: Applied the edits.' in out
        assert 'OBSERVATION: Both hunks landed.' in out

    def test_the_routing_marker_survives(self):
        state = pointed_state()
        state.build(CONTEXT)

        out = state.strip_pinned(2, REPORT)

        assert out.rstrip().endswith('READY_FOR_REVIEW')

    def test_it_actually_gets_shorter(self):
        state = pointed_state()
        state.build(CONTEXT)

        # PLAN is where the long pinned section lives.
        assert len(state.strip_pinned(1, PLAN)) < len(PLAN)

    def test_a_body_shorter_than_the_pointer_is_left_alone(self):
        """Pointing "REMAINING: none" at the block is strictly worse."""
        state = pointed_state()
        state.build(CONTEXT)

        out = state.strip_pinned(2, REPORT)

        assert 'REMAINING: none' in out
        assert 'see REMAINING in' not in out

    def test_nothing_is_stripped_before_a_block_is_built(self):
        # The pointer would refer to a block that does not exist.
        assert pointed_state().strip_pinned(2, REPORT) == REPORT

    def test_only_the_entry_the_block_quoted_is_stripped(self):
        """`latest_only` means an older copy is the only record of its wording."""
        older = (
            'an earlier concern nobody restated, long enough that pointing at '
            'the block would genuinely be shorter than leaving it here.'
        )
        earlier = LONG_REPORT.replace(LONG_CONCERNS, older)
        state = pointed_state()
        state.build(['issue', earlier, LONG_REPORT])

        # turn 3 is what the block quotes, so turn 2 keeps its own wording
        assert older in state.strip_pinned(1, earlier)
        assert LONG_CONCERNS not in state.strip_pinned(2, LONG_REPORT)

    def test_the_block_still_carries_what_the_body_gave_up(self):
        state = pointed_state()
        built = state.build(CONTEXT)

        assert 'the fallback at line 233 must never be reached' in built
        assert 'wrap_var is only called from task_executor.py' in built

    def test_without_pointers_both_copies_remain(self):
        state = ResumptionState(policy())
        state.build(CONTEXT)

        assert state.strip_pinned(2, REPORT) == REPORT
