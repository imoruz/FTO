"""Needle test: the gate on a compression rate, run against the real model.

LLMLingua-2 is task-agnostic. It scores relevance uniformly and knows nothing
about what the restarted node is about to do, so nothing in the pipeline
notices when a rate is too aggressive for the content -- the only protections
are ``force_tokens``, ``PROTECTED_SPANS`` and ``min_fragment_chars``. This
plants facts a restarted agent would need, compresses, and checks they came
back, so a rate is never trusted on the strength of its ratio alone.

Skipped unless ``FTO_NEEDLE_TEST=1``: it downloads and runs a ~2GB scoring
model, which has no business in the unit suite.

    FTO_NEEDLE_TEST=1 pytest tests/recovery/test_needle.py -v -s

Needles must be strings that do *not* already occur in the haystack, or a
survival check passes on a pre-existing copy and proves nothing.

One limit this cannot assert as an invariant, but which is worth knowing: the
guarantee is by backtick, not by looking like code. A literal written as bare
prose goes through the scorer like any other words. Whether it survives is
content-dependent -- in a synthetic haystack `force_reserve_digit` carried
``cpe:2.9:x:acme:widgetry`` through at rate 0.4, while in a real Planner plan
``cpe:2.3:o:fortinet`` came back as ``:2.3`` at the same rate. So a MAS whose
prompts need literals preserved has to quote them or name them in
``force_tokens``; do not rely on the digit heuristic.
"""

import os

import pytest

from fto.recovery.compression import (
    DEFAULT_FORCE_TOKENS,
    LLMLinguaCompressor,
    SectionPolicy,
    StructuredCompressor,
)

pytestmark = pytest.mark.skipif(
    os.environ.get('FTO_NEEDLE_TEST') != '1',
    reason='loads a ~2GB scoring model; set FTO_NEEDLE_TEST=1 to run',
)

# The MAS-specific markers an experiment adds on top of fto's defaults. Kept
# here rather than imported so the gate does not depend on the experiment repo.
PLANNER_CODER_FORCE_TOKENS = [
    *DEFAULT_FORCE_TOKENS,
    'TASK_COMPLETE',
    'READY_FOR_REVIEW',
    'EDITS',
    'PLAN_DEVIATIONS',
    'CONCERNS',
    'REMAINING',
    'THOUGHT',
    'ACTION',
    'OBSERVATION',
    'REPORT',
    'OUTPUT',
]

# Filler with no needle in it, long enough that the compressor has something
# to throw away and the needles are not simply the whole input.
FILLER = (
    'The converter walks the parsed device tables and assembles identifiers '
    'for every product line it recognises, appending one entry per identity '
    'it can derive and skipping the rest. Earlier revisions of this routine '
    'handled a single vendor family, so the surrounding helpers assume a '
    'flat shape and the caller deduplicates afterwards rather than during '
    'assembly. Reviewers have asked before whether the ordering matters; it '
    'does not, because the caller sorts before comparing. '
) * 3

# Each needle is a fact a restarted reviewer would be wrong without.
NEEDLES = {
    'backticked path': '`internal/detect/vendorlist.go`',
    'backticked literal': '`cpe:2.9:x:acme:widgetry`',
    'backticked version': '`v11.7.3`',
    'bare line range': 'lines 412-418',
    'negation': 'must never',
    'bare identifier': 'VendorPhysicalTag',
}

HAYSTACK = (
    f'{FILLER}\n\n'
    f'## Assumptions & Open Questions\n'
    f'- The hardware identity comes from {NEEDLES["bare identifier"]} and '
    f'nowhere else, so a device missing that field is out of scope here.\n'
    f'- Edge case to watch: {NEEDLES["backticked path"]} at '
    f'{NEEDLES["bare line range"]} {NEEDLES["negation"]} emit '
    f'{NEEDLES["backticked literal"]} for a switch reporting '
    f'{NEEDLES["backticked version"]}, because that identifier belongs to the '
    f'router family and the switch family has its own.\n'
    f'{FILLER}'
)


def _compressor(rate):
    return LLMLinguaCompressor(rate=rate, force_tokens=PLANNER_CODER_FORCE_TOKENS)


def _survivors(rate):
    compressed = _compressor(rate).compress([HAYSTACK]).texts[0]
    return {name: needle in compressed for name, needle in NEEDLES.items()}


@pytest.fixture(scope='module')
def needles_are_unique():
    for name, needle in NEEDLES.items():
        assert HAYSTACK.count(needle) == 1, f'{name} must occur exactly once'


@pytest.mark.parametrize('rate', [0.55, 0.4])
def test_the_configured_rate_keeps_every_needle(rate, needles_are_unique):
    """The rates an experiment should actually run at."""
    survived = _survivors(rate)
    lost = [name for name, kept in survived.items() if not kept]
    assert not lost, f'rate {rate} lost: {lost}'


def test_compression_actually_happened():
    """Guard against a vacuous pass: needles surviving an untouched string."""
    result = _compressor(0.55).compress([HAYSTACK])

    assert result.rate < 0.95, f'nothing was compressed ({result.describe()})'
    assert len(result.texts[0]) < len(HAYSTACK)


def test_code_spans_survive_whatever_the_rate():
    """Protected spans bypass the model, so the rate cannot touch them."""
    for rate in (0.55, 0.25, 0.1):
        survived = _survivors(rate)
        for name in ('backticked path', 'backticked literal', 'backticked version'):
            assert survived[name], f'{name} lost at rate {rate}'


def test_negations_survive_whatever_the_rate():
    """force_tokens pins them, so pruning cannot invert an instruction."""
    for rate in (0.55, 0.25, 0.1):
        assert _survivors(rate)['negation'], f'negation lost at rate {rate}'


def test_a_short_control_marker_entry_is_kept_verbatim():
    """`min_fragment_chars` covers a marker that is a message of its own."""
    result = _compressor(0.55).compress(['TASK_COMPLETE', 'READY_FOR_REVIEW'])

    assert result.texts == ['TASK_COMPLETE', 'READY_FOR_REVIEW']


def test_control_markers_buried_in_a_long_report_survive_too():
    """Inside a long message the threshold does not apply -- force_tokens do."""
    report = (
        'THOUGHT: I implemented the plan and checked each edit against the file '
        'to make sure the hunks landed where the plan said they should.\n\n'
        'ACTION: Applied the edits and re-read every region afterwards.\n\n'
        'OBSERVATION: All three hunks landed where intended and nothing else '
        'in the surrounding code was disturbed by the change.\n\n'
        'REPORT:\nEDITS:\n- reworked the vendor branch to handle the new tag\n'
        'PLAN_DEVIATIONS: none\n\nCONCERNS: none\n\nREMAINING: none\n\n'
        'READY_FOR_REVIEW'
    )
    compressed = _compressor(0.55).compress([report]).texts[0]

    assert len(compressed) < len(report), 'the report should have been compressed'
    for marker in (
        'READY_FOR_REVIEW',
        'CONCERNS',
        'PLAN_DEVIATIONS',
        'REMAINING',
        'none',
    ):
        assert marker in compressed, f'{marker} did not survive'


# --- structure-aware compression of the newest message ----------------------

PLANNER_CODER_SECTIONS = [
    SectionPolicy('THOUGHT', 0.3),
    SectionPolicy('ACTION', 0.9),
    SectionPolicy('OBSERVATION', 0.5),
    SectionPolicy('REPORT', 0.9),
    SectionPolicy('EDITS', 0.9),
    SectionPolicy('PLAN_DEVIATIONS', 0.9),
    SectionPolicy('CONCERNS', None),
    SectionPolicy('REMAINING', None),
]

CODER_REPORT = (
    'THOUGHT: The plan asked me to widen the vendor branch so the switch '
    'family is recognised as well as the router family, and I checked the '
    'existing helpers still cover what the plan assumed about them.\n\n'
    'ACTION: Applied the edits with the file tools and re-read each region '
    'afterwards to confirm the hunks landed where they were meant to.\n\n'
    'OBSERVATION: All three hunks landed where intended and the surrounding '
    'code is untouched, and the helper still returns the same shape.\n\n'
    'REPORT:\n'
    'EDITS:           `internal/detect/vendorlist.go` at lines 412-418, '
    'widening the prefix table and adding the switch family mapping.\n'
    'PLAN_DEVIATIONS: none, the plan was followed exactly as written.\n\n'
    'CONCERNS:        the fallback path at line 233 must never be reached '
    'for a switch, because it assumes the router identifier.\n\n'
    'REMAINING: none\n\n'
    'READY_FOR_REVIEW'
)


def _structured():
    return StructuredCompressor(
        sections=PLANNER_CODER_SECTIONS,
        template=_compressor(0.55),
    )


def test_the_scaffold_survives_structured_compression():
    """The next agent's prompt tells it to read these fields by name."""
    out = _structured().compress([CODER_REPORT]).texts[0]

    assert [label.strip() for label in _structured().labels_of(out)] == [
        'THOUGHT:', 'ACTION:', 'OBSERVATION:', 'REPORT:',
        'EDITS:', 'PLAN_DEVIATIONS:', 'CONCERNS:', 'REMAINING:',
    ]


def test_a_verbatim_section_is_byte_identical():
    out = _structured().compress([CODER_REPORT]).texts[0]

    assert 'the fallback path at line 233 must never be reached' in out
    assert 'REMAINING: none' in out


def test_narration_is_compressed_harder_than_the_record():
    sc = _structured()
    before = {label.strip(): body for label, body, _ in sc._sections(CODER_REPORT)}
    after = {
        label.strip(): body
        for label, body, _ in sc._sections(sc.compress([CODER_REPORT]).texts[0])
    }

    thought = len(after['THOUGHT:']) / len(before['THOUGHT:'])
    edits = len(after['EDITS:']) / len(before['EDITS:'])
    assert thought < edits, f'THOUGHT {thought:.0%} should shrink more than EDITS {edits:.0%}'


def test_the_message_actually_gets_shorter():
    result = _structured().compress([CODER_REPORT])

    assert result.compressed_tokens < result.origin_tokens
    assert len(result.texts[0]) < len(CODER_REPORT)


def test_the_terminal_marker_survives():
    assert 'READY_FOR_REVIEW' in _structured().compress([CODER_REPORT]).texts[0]


def test_needles_inside_a_compressed_section_survive():
    out = _structured().compress([CODER_REPORT]).texts[0]

    assert '`internal/detect/vendorlist.go`' in out   # protected span
    assert 'lines 412-418' in out                     # digit-bearing
    assert 'none' in out                              # pinned by force_tokens
