"""The pinned resumption state block a restarted node is rebuilt on top of.

Restarting an agent is not replaying a chat log, it is writing a
reinstantiation prompt, and a few fields in that prompt are *state* rather
than prose. In the Planner-Coder loop three of them cross the agent boundary
and are load-bearing:

- the Planner's ``Assumptions & Open Questions``, which the Coder's first
  implementation step tells it to validate one by one;
- the Coder's ``CONCERNS`` and ``PLAN_DEVIATIONS``, which the Planner's review
  steps require a verdict on -- and ``CONCERNS`` gates ``TASK_COMPLETE``;
- the ledger of what has actually been edited, which the review verifies with
  ``read_file_segment`` against exact line numbers.

Lose a concern to token pruning and the Planner can emit ``TASK_COMPLETE``
early. That is a correctness failure with no error signal, which is precisely
the kind this loop cannot detect on its own. So these fields are lifted out of
the history, restated verbatim at the top of the rebuilt prompt, and never
shown to a compressor. Nothing here calls a model: the block is regenerated
structurally from the snapshot.

It also carries the resumption marker itself. Both role prompts open with a
THOUGHT that asks the agent to place itself in the loop -- "state whether this
is your first turn", "what was already done in prior iterations?" -- so a
restarted agent handed an unmarked context concludes it is turn one, re-plans
from scratch and discards the work it is resuming.

Position is deliberate. The block sits directly after the frozen zone and
before the compressed history, so the two highest-value parts of the prompt
(this block and the newest message) sit at the two ends rather than in the
middle.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from fto.recovery.compression import (
    LINE_REF_SPANS,
    PATH_SPANS,
    PROTECTED_SPANS,
    build_label_pattern,
    normalise_label,
)

DEFAULT_HEADING = '── RESUMPTION STATE ──'
DEFAULT_AUTHORITY_NOTE = (
    'The state block above is authoritative. Where the compressed history '
    'below conflicts with it, trust this block.'
)


@dataclass
class PinnedField:
    """One labelled section lifted into the state block.

    ``label`` is matched the same way ``StructuredCompressor`` matches it, so
    the two agree on where a section starts. ``title`` is how it is announced
    in the block -- the spec's wording ("OPEN CONCERNS") rather than the
    agent's raw label, because the block is telling the agent what is still
    outstanding, not quoting it back.

    ``ledger`` renders the section as ``path -> line refs`` per turn instead
    of as prose: that is the shape the Planner's review step actually uses,
    and it survives even when the surrounding description does not.
    """

    label: str
    title: str
    #: Replace this section's body in the message with this line, instead of
    #: leaving a second verbatim copy behind the block. ``{title}`` is
    #: substituted. None keeps both copies.
    pointer: str | None = None
    ledger: bool = False
    #: Keep only the most recent turn's value. The Coder restates CONCERNS
    #: and PLAN_DEVIATIONS in full every turn, so the latest one supersedes;
    #: an edit ledger accumulates instead.
    latest_only: bool = True


@dataclass
class ResumptionPolicy:
    """What the state block restates, and how it is framed.

    Which labels matter is a property of the target MAS's prompts, so the
    fields come from configuration -- either directly, or read off the
    ``pin`` titles of the section policies the structured compressor already
    carries (``from_sections``).
    """

    fields: List[PinnedField] = field(default_factory=list)
    #: Every label the message may carry, pinned or not. A pinned section
    #: ends where the *next* section starts, so without the unpinned labels
    #: CONCERNS swallows REMAINING and the trailing routing marker with it.
    boundary_labels: List[str] = field(default_factory=list)
    heading: str = DEFAULT_HEADING
    authority_note: str = DEFAULT_AUTHORITY_NOTE
    #: Routing keywords an agent appends after its last field. They belong to
    #: the message, not to the field they happen to trail, and a state block
    #: that quotes one is both noise and a hazard -- so they are stripped off
    #: the end of a harvested body.
    trailing_markers: List[str] = field(default_factory=list)
    #: Label whose presence marks a message as a turn of the loop-counted
    #: agent, for the LoopGuard line. None omits the line.
    turn_marker_label: str | None = None
    #: The loop counter's ``max_iterations``. None omits the line.
    loop_guard_limit: int | None = None

    @classmethod
    def from_sections(cls, sections, **kwargs) -> 'ResumptionPolicy':
        """Read the pinned fields off ``SectionPolicy.pin`` titles."""
        return cls(
            fields=[
                PinnedField(
                    label=s.label,
                    title=s.pin,
                    pointer=getattr(s, 'pin_pointer', None),
                    ledger=normalise_label(s.label) == 'edits',
                    latest_only=normalise_label(s.label) != 'edits',
                )
                for s in sections
                if getattr(s, 'pin', None)
            ],
            boundary_labels=[s.label for s in sections],
            **kwargs,
        )


class ResumptionState:
    """Builds the pinned block from the snapshot, without calling a model."""

    def __init__(self, policy: ResumptionPolicy) -> None:
        self.policy = policy
        #: (entry index, pin title) pairs the last ``build`` actually rendered.
        #: Only these may be replaced by a pointer -- a section that was
        #: harvested but superseded by a later turn is still the only copy of
        #: its own wording.
        self.rendered: set = set()
        self._by_key = {normalise_label(f.label): f for f in policy.fields}
        labels = list(policy.boundary_labels) or [f.label for f in policy.fields]
        for pinned in policy.fields:
            if pinned.label not in labels:
                labels.append(pinned.label)
        self._pattern = build_label_pattern(labels)

    def build(self, texts: List[str]) -> str:
        """The block for a context, or '' when there is no state to pin.

        ``texts`` is the whole snapshot in order, uncompressed -- the block is
        a restatement of the original wording, which is the entire point of
        holding these fields outside the compressor.
        """
        if not self.policy.fields or not any(texts):
            return ''

        found = self._harvest(texts)
        self.rendered = set()
        lines = [self.policy.heading, self._marker(texts)]
        loop_guard = self._loop_guard(texts)
        if loop_guard:
            lines.append(loop_guard)

        body = [
            rendered
            for pinned in self.policy.fields
            for rendered in self._render(pinned, found.get(pinned.title, []))
        ]
        if not body:
            return ''
        lines += body
        lines.append(self.policy.authority_note)
        return '\n'.join(lines)

    def strip_pinned(self, index: int, text: str) -> str:
        """Replace pinned bodies in one entry with their pointer line.

        Only sections this block actually rendered from *this* entry are
        touched, and only those configured with a ``pointer``. The label and
        the surrounding whitespace stay, so the message keeps its shape and
        the agent still finds the field; what goes is the second verbatim
        copy of text the block already carries.

        Call it only when ``build`` returned a block, or the pointer dangles.
        """
        if self._pattern is None or not text or not self.rendered:
            return text

        matches = list(self._pattern.finditer(text))
        out, at = [], 0
        for k, match in enumerate(matches):
            pinned = self._by_key.get(normalise_label(match.group(0)))
            if pinned is None or not pinned.pointer:
                continue
            if (index, pinned.title) not in self.rendered:
                continue
            end = matches[k + 1].start() if k + 1 < len(matches) else len(text)
            body = text[match.end() : end]
            kept = _trailing_markers_of(body, self.policy.trailing_markers)
            stripped = _strip_markers(body, self.policy.trailing_markers)
            if not stripped:
                continue
            # A pointer is only worth it when it is shorter than what it
            # replaces. "REMAINING: none" pointed at the block is strictly
            # worse than "REMAINING: none", and no configuration should be
            # able to make that trade by accident.
            pointer = pinned.pointer.format(title=pinned.title)
            if len(pointer) >= len(stripped.strip()):
                continue
            # Keep the body's own leading break, so a `## Heading` label --
            # whose match stops at the end of its line -- does not end up
            # welded to the pointer, while a `LABEL: ` one stays inline.
            lead = body[: len(body) - len(body.lstrip('\n'))]
            trail = body[len(body.rstrip()) :]
            out.append(text[at : match.end()])
            out.append(lead + pinned.pointer.format(title=pinned.title) + kept + trail)
            at = end
        if not out:
            return text
        out.append(text[at:])
        return ''.join(out)

    def _harvest(self, texts: List[str]) -> Dict[str, List[Tuple[int, str]]]:
        """Every pinned section body, as ``title -> [(turn, body), ...]``."""
        found: Dict[str, List[Tuple[int, str]]] = {}
        if self._pattern is None:
            return found
        for index, text in enumerate(texts):
            if not text:
                continue
            matches = list(self._pattern.finditer(text))
            for k, match in enumerate(matches):
                pinned = self._by_key.get(normalise_label(match.group(0)))
                if pinned is None:
                    continue
                end = matches[k + 1].start() if k + 1 < len(matches) else len(text)
                body = _strip_markers(
                    text[match.end() : end], self.policy.trailing_markers
                )
                if body:
                    found.setdefault(pinned.title, []).append((index + 1, body))
        return found

    def _render(self, pinned: PinnedField, hits: List[Tuple[int, str]]) -> List[str]:
        if not hits:
            return []
        if pinned.ledger:
            return self._render_ledger(pinned, hits)
        # Both agents restate these fields in full every turn, so the latest
        # statement supersedes the rest.
        chosen = hits[-1:] if pinned.latest_only else hits
        # turn numbers are 1-based; the entry index is what strip_pinned keys on
        self.rendered.update((turn - 1, pinned.title) for turn, _ in chosen)
        return [
            line
            for turn, body in chosen
            for line in [f'{pinned.title} (turn {turn}):', *_indent(body)]
        ]

    @staticmethod
    def _render_ledger(pinned: PinnedField, hits: List[Tuple[int, str]]) -> List[str]:
        """``path -> line refs`` per turn, harvested structurally.

        The descriptions around an edit are prose and compress; the file and
        the lines are what the review reads back, so they are extracted
        rather than quoted.
        """
        lines = [f'{pinned.title}:']
        for turn, body in hits:
            for path, refs in _ledger_entries(body):
                refs_text = ', '.join(refs) if refs else 'lines not stated'
                lines.append(f'  {path} -> {refs_text}   [turn {turn}]')
        return lines if len(lines) > 1 else []

    def _marker(self, texts: List[str]) -> str:
        turn = sum(1 for text in texts if text.strip()) or 1
        if turn <= 1:
            return (
                'Resumed after a fault. Your previous attempt at this turn was '
                'discarded; the context below is what you were handed, not what '
                'you produced.'
            )
        return (
            f'Resumed after a fault. This is turn {turn}, not your first turn: '
            f'work from what is already recorded below rather than planning '
            f'from scratch.'
        )

    def _loop_guard(self, texts: List[str]) -> str:
        label = self.policy.turn_marker_label
        limit = self.policy.loop_guard_limit
        if not label or not limit:
            return ''
        pattern = build_label_pattern([label])
        consumed = sum(1 for text in texts if text and pattern.search(text))
        return (
            f'LoopGuard: {consumed}/{limit} counted turns consumed. A restart '
            f"does not consume another: the faulty attempt's outgoing edges "
            f'are withheld and only the successful attempt is released.'
        )


def _trailing_markers_of(body: str, markers: List[str]) -> str:
    """The routing markers trailing a body, to re-attach after a pointer.

    ``READY_FOR_REVIEW`` sits after the Coder's last field but belongs to the
    message, not the field -- dropping it with the body would break routing.
    """
    kept = []
    rest = body.rstrip()
    for marker in markers:
        if rest.endswith(marker):
            kept.insert(0, marker)
            rest = rest[: -len(marker)].rstrip()
    return ('\n\n' + '\n\n'.join(kept)) if kept else ''


def _strip_markers(body: str, markers: List[str]) -> str:
    """Drop a routing keyword trailing the last field of a message."""
    body = body.strip()
    changed = True
    while changed and markers:
        changed = False
        for marker in markers:
            if body.endswith(marker):
                body = body[: -len(marker)].strip()
                changed = True
    return body


def _indent(body: str) -> List[str]:
    return ['  ' + line if line.strip() else line for line in body.splitlines()]


def _ledger_entries(body: str) -> List[Tuple[str, List[str]]]:
    """``(path, [line refs])`` for each file named in an EDITS body.

    A line reference belongs to the path most recently named before it, which
    is how these reports are actually written ("`x.py` at lines 412-418").
    """
    entries: List[Tuple[str, List[str]]] = []
    seen: Dict[str, List[str]] = {}
    order: List[str] = []
    for match in re.finditer(
        f'(?P<path>{PATH_SPANS.pattern})|(?P<ref>{LINE_REF_SPANS.pattern})',
        _unquote(body),
        re.IGNORECASE,
    ):
        if match.group('path'):
            path = match.group('path')
            if path not in seen:
                seen[path] = []
                order.append(path)
            current = path
        elif order:
            current = order[-1]
            ref = match.group('ref')
            if ref not in seen[current]:
                seen[current].append(ref)
    entries = [(path, seen[path]) for path in order]
    return entries


def _unquote(text: str) -> str:
    """Strip backticks so a quoted path is matched like a bare one."""
    return PROTECTED_SPANS.sub(lambda m: m.group().strip('`'), text)
