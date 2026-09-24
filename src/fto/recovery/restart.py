from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List

from dataclasses import dataclass, field

from fto.recovery.compression import (
    CompressionFailure,
    CompressionResult,
    CompressionValidator,
    ContextCompressor,
    LLMLinguaCompressor,
    guard_control_literals,
    keep_labelled_sections,
    build_label_pattern
)
from fto.recovery.resumption import ResumptionPolicy, ResumptionState

if TYPE_CHECKING:
    from fto.adapters.node.node import NodeAdapter

# Separates the pinned resumption state block from the (compressed) message
# that follows it, so the two are never mistaken for one continuous block --
# the state block is regenerated structurally, the text under it is what the
# node actually queued.
RESUMPTION_BOUNDARY = '── COMPRESSED HISTORY FROM THE PREVIOUS ATTEMPT ──'

# Separates the compressed history from the newest message(s), when those are
# run through active_compressor: the two are compressed by different means
# (token-pruned prose vs. section-aware structural compression) and marking
# the seam keeps a reader -- human or model -- from reading them as one
# continuous, uniformly-compressed block.
TAIL_BOUNDARY = '── STRUCTURALLY COMPRESSED LATEST MESSAGE ──'


@dataclass
class MessageRecord:
    """What happened to one message, for auditing a run after the fact.

    A ``verbatim`` policy with ``mutated=True`` is a bug, not a nuance -- the
    whole point of the protected head and tail is that they come back
    untouched.
    """

    index: int
    policy: str
    chars_before: int
    chars_after: int
    failures: List[str] = field(default_factory=list)

    @property
    def mutated(self) -> bool:
        return self.chars_before != self.chars_after


def _merge(
    history: CompressionResult, active: CompressionResult | None
) -> CompressionResult:
    """One result covering both zones, so the logged ratio is the real one."""
    if active is None:
        return history
    return CompressionResult(
        history.texts + active.texts,
        origin_tokens=history.origin_tokens + active.origin_tokens,
        compressed_tokens=history.compressed_tokens + active.compressed_tokens,
    )


class Restart:
    def __init__(self, restart_count: int = 1) -> None:
        # Maximum number of times a single node may be restarted
        self.restart_count = restart_count
        self.context: Any = None
        self.adapter: 'NodeAdapter | None' = None

    def set_context(self, context: Any, adapter: 'NodeAdapter | None' = None) -> None:
        """Snapshot the node's context as it stood before the faulty execution.

        ``adapter`` is the node adapter that context was read from. Restart
        with refined context needs it to read the context as text and to put
        the refined text back into MAS-native messages.
        """
        self.context = deepcopy(context)
        self.adapter = adapter

    def set_diff(self, diff_path: Path | None) -> None:
        pass

    def set_idx(self, idx: int) -> None:
        self.idx = idx

    def get_context(self) -> Any:
        pass

    def restore_input(self, adapter: 'NodeAdapter', injected_context: Any) -> None:
        """Undo a one-time context override once the retried execution has run.

        A no-op by default: most modes mean ``get_context()``'s result to
        become the node's new *permanent* history -- that's the whole point
        of a reset (``RestartAllContext``) or a compressed history
        (``RestartRefinedContext``), and the node's transcript is meant to
        keep building on top of it. Only a mode whose context is a one-shot
        engineered message for a single retried call (``RestartWithDiff``,
        ``RestartWithDiffSections``) needs to splice that message back out
        afterward, or it becomes a permanent, stale fixture of every future
        turn -- see ``Manager._restart``, which calls this after each attempt.
        """
        pass


class RestartAllContext(Restart):
    def __init__(self, restart_count: int = 1) -> None:
        super().__init__(restart_count)

    def get_context(self) -> Any:
        return self.context


class RestartNoContext(Restart):
    def __init__(self, restart_count: int = 1) -> None:
        super().__init__(restart_count)

    def get_context(self) -> Any:
        return []


class RestartRefinedContext(Restart):
    """Restart the node on a compressed history plus its latest message intact.

    Every message the node had queued is compressed except the last
    ``keep_last``: those are what the node is being asked to act on right now,
    so they go back in verbatim. A Coder restarted mid-implementation, for
    instance, comes back to a compressed plan and a compressed copy of its own
    earlier report, but to the review it still has to answer in full.

    Compression rewrites only the text inside a message. Roles, sources, keep
    flags and attachments are preserved, so the refined context stays a valid
    input for the same node.
    """

    def __init__(
        self,
        restart_count: int = 1,
        compressor: ContextCompressor | None = None,
        keep_last: int = 1,
        logger=None,
        on_error: CompressionFailure | str = CompressionFailure.RAISE,
        keep_first: int = 1,
        validator: CompressionValidator | None = None,
        min_chars: int = 0,
        min_saving: float = 0.0,
        active_compressor: ContextCompressor | None = None,
        resumption: ResumptionPolicy | ResumptionState | None = None,
        control_literals: List[str] | None = None,
        history_sentinel: str = '[history]',
    ) -> None:
        super().__init__(restart_count)
        self.compressor = (
            compressor if compressor is not None else LLMLinguaCompressor()
        )
        # At least one message has to survive intact, or the node is restarted
        # with no statement of what it is being asked to do.
        self.keep_last = max(1, keep_last)
        # The oldest messages are the instruction-like ones -- the task
        # statement for a planner, the original plan for a coder -- and
        # instructions are the content token pruning damages worst. Upstream's
        # own guidance separates instructions from context for exactly this
        # reason, so they are held out rather than pruned.
        self.keep_first = max(0, keep_first)
        self.validator = validator if validator is not None else CompressionValidator()
        # Don't pay semantic risk for a saving that isn't there. Both gates are
        # off by default so an existing config keeps its behaviour; see
        # docs/compression.md for the measured case for turning them on.
        self.min_chars = max(0, min_chars)
        self.min_saving = min_saving
        # What to do with the newest message(s). None keeps them verbatim,
        # which is the safe default: they are what the node has to act on.
        # Give it a StructuredCompressor to shrink them *along their own
        # labels* instead -- the scaffold the next agent reads survives, and
        # each field is compressed by what it is worth.
        self.active_compressor = active_compressor
        # The pinned state block: assumptions, concerns, deviations and the
        # edit ledger, restated verbatim above the compressed history so a
        # dropped concern cannot let the loop declare itself finished. None
        # disables it; a MAS that has no such fields has nothing to pin.
        self.resumption = (
            ResumptionState(resumption)
            if isinstance(resumption, ResumptionPolicy)
            else resumption
        )
        # Keywords the workflow's edges match on. A compressed history entry
        # that happens to *end* with one satisfies a terminal regex without
        # any agent having decided anything, so it gets a sentinel line.
        self.control_literals = list(control_literals or [])
        self.history_sentinel = history_sentinel
        self.records: List[MessageRecord] = []
        self.logger = logger
        self.on_error = CompressionFailure(on_error)
        self.last_result: CompressionResult | None = None
        #: Did compression actually run and succeed for this snapshot? None
        #: before the first attempt. Read it per turn so a benchmark never
        #: contains an invisible mix of refined and unrefined restarts.
        self.compressed: bool | None = None
        #: Why the history was not compressed, when it was not.
        self.failure: str | None = None
        #: The pinned state block prepended to this snapshot's context, for
        #: auditing what the restarted node was actually told.
        self.pinned_state: str | None = None
        self._texts: List[str] | None = None

    def set_context(self, context: Any, adapter: 'NodeAdapter | None' = None) -> None:
        super().set_context(context, adapter=adapter)
        self._texts = None
        self.last_result = None
        self.compressed = None
        self.failure = None
        self.pinned_state = None
        self.records = []

    def get_context(self) -> Any:
        if not self.context:
            return self.context
        if self.adapter is None:
            raise ValueError(
                'RestartRefinedContext needs the node adapter to read and '
                'rebuild the context; pass it to set_context().'
            )

        texts = self._refine()
        if texts is None:
            return self.context
        # Rebuilt fresh every time: a restarted node is free to mutate the
        # input it was handed, and the next attempt must not inherit that.
        return self.adapter.context_from_list(texts, self.context)

    def refine_test(self, texts: str = None) -> List[str] | None:
        """Test-only alias for ``_refine``; kept public so callers can inspect it.

        There is no separate logic here on purpose -- a duplicate copy of
        ``_refine`` drifted out of sync with it (missing the tail boundary
        marker, still calling the now-removed history validation) and produced
        different output than the real restart path. This delegates so the
        two can never diverge again.
        """
        return self._refine(texts)

    def _refine(self, texts: str = None) -> List[str] | None:
        """The refined text of every message, or None if there is nothing to do.

        Compression runs once per snapshot: further attempts on the same node
        replay the same refined context instead of paying for the model again.
        """
        if self._texts is not None:
            return self._texts
        texts = texts or self.adapter.context_as_list(self.context)


        # TODO: TEST FROM HERE WHAT HAPPENS


        head, history, latest = self._split(texts)

        # Everything that could be compressed: the history always, and the
        # tail too when a compressor was given for it.
        payload = [text for text in history if text.strip()]
        if self.active_compressor is not None:
            payload += [text for text in latest if text.strip()]

        if not payload:
            return self._skip(
                head,
                history,
                latest,
                texts,
                'nothing to compress outside the protected head'
                + ('' if self.active_compressor else ' and the verbatim tail'),
            )
        if sum(len(text) for text in payload) < self.min_chars:
            return self._skip(
                head,
                history,
                latest,
                texts,
                f'only {sum(len(t) for t in payload)} chars to compress, under '
                f'min_chars ({self.min_chars}); not worth the semantic risk',
            )

        result = (
            self._compress(history, latest)
            if any(text.strip() for text in history)
            else CompressionResult([''] * len(history))
        )
        active = self._compress_active(latest)
        combined = _merge(result, active)
        if self.min_saving and combined.rate > 1 - self.min_saving:
            skipped = self._skip(
                head,
                history,
                latest,
                texts,
                f'saving {1 - combined.rate:.1%} '
                f'below min_saving ({self.min_saving:.0%})',
            )
            self.last_result = combined
            return skipped

        # Validation against negations/code-spans/identifiers was dropped: it
        # rejected too many faithful compressions in practice (see the
        # StructuredCompressor sections, which are pinned or dropped rather
        # than checked after the fact). Compressor output is accepted as-is.
        refined = result.texts
        tail = active.texts if active is not None else [''] * len(latest)
        rejected = [[] for _ in history]
        tail_rejected = [[] for _ in latest]
        self.last_result = combined
        self._record(head, history, refined, rejected, latest, tail, tail_rejected)
        self._log(
            combined,
            head=head,
            history=history,
            latest=latest,
            rejected=rejected + tail_rejected,
        )

        # '' means "leave that message as it was", which is what the protected
        # head needs -- and the right thing to do for an entry that compressed
        # down to nothing or was rejected by the validator.
        # A fallback or a failed tail already recorded why this turn is not
        # refined; do not overwrite that.
        if self.failure is None:
            self.compressed = True

        refined = [
            guard_control_literals(text, self.control_literals, self.history_sentinel)
            if text
            else text
            for text in refined
        ]
        combined_texts = [''] * len(head) + refined + tail
        if active is not None and any(t.strip() for t in history):
            combined_texts = self._with_tail_boundary(
                combined_texts, len(head) + len(history)
            )
        self._texts = self._with_resumption(combined_texts, texts, len(head))
        return self._texts

    def _with_tail_boundary(
        self, refined: List[str], tail_start: int
    ) -> List[str]:
        """Mark the seam between the compressed history and the tail zone.

        Only meaningful when the tail actually went through
        ``active_compressor`` *and* there is compressed history in front of
        it to be mistaken for -- a verbatim tail, or a tail with nothing
        ahead of it, needs no marker.
        """
        for index in range(tail_start, len(refined)):
            if not refined[index].strip():
                continue
            refined[index] = f'{TAIL_BOUNDARY}\n\n{refined[index]}'
            return refined
        return refined

    def _with_resumption(
        self, refined: List[str], originals: List[str], head: int
    ) -> List[str]:
        """Prepend the pinned state block to the first entry after the head.

        The head is the frozen zone -- the task statement, passed through
        verbatim -- so the block goes immediately after it and before the
        compressed history, which is where the spec puts it and where a model
        actually attends to it. Built from ``originals``: the whole point of
        these fields is that no compressor ever sees them.
        """
        if self.resumption is None:
            return refined
        block = self.resumption.build(originals)
        if not block:
            return refined

        # The block now carries these fields verbatim, so any section
        # configured with a pointer gives up its second copy. Done only once
        # the block exists, or the pointer would refer to nothing.
        for index in range(head, len(refined)):
            current = refined[index] or originals[index]
            pointed = self.resumption.strip_pinned(index, current)
            if pointed != current:
                refined[index] = pointed
                if index < len(self.records):
                    self.records[index].chars_after = len(pointed)

        for index in range(head, len(refined)):
            current = refined[index] or originals[index]
            if not current.strip():
                continue
            refined[index] = f'{block}\n\n{RESUMPTION_BOUNDARY}\n\n{current}'
            self.pinned_state = block
            if index < len(self.records):
                self.records[index].chars_after = len(refined[index])
            return refined
        return refined

    def _compress_active(self, latest: List[str]) -> CompressionResult | None:
        """Compress the newest message(s), if a compressor was given for them.

        None means the tail is kept verbatim, which is the default: it is what
        the node has to act on. A ``StructuredCompressor`` here shrinks it
        along its own labels instead, so the scaffold the next agent reads
        stays intact and each field is compressed by what it is worth.
        """
        if self.active_compressor is None or not any(t.strip() for t in latest):
            return None
        try:
            return self.active_compressor.compress(latest)
        except Exception as exc:
            self.compressed = False
            self.failure = f'{type(exc).__name__}: {exc}'
            if self.on_error is CompressionFailure.RAISE:
                raise
            self._warn(
                f'Refined context: structured compression of the newest '
                f'message failed ({self.failure}); keeping it whole. This '
                f'turn is NOT refined -- exclude it when comparing modes.'
            )
            return None

    def _split(self, texts: List[str]) -> tuple[List[str], List[str], List[str]]:
        """Protected head, compressible history, active tail.

        The tail is allocated first. When a node is restarted on its very
        first turn its whole context is one message -- the plan a Coder was
        just handed -- and that message is simultaneously the oldest and the
        newest. It is what the node has to act on, so the tail claims it;
        giving the head priority instead leaves the tail empty and nothing
        happens at all, which is what a first-turn restart used to do.

        For any context of two or more messages this is the same split as
        head-first, so only that degenerate case changes.
        """
        last = min(self.keep_last, len(texts))
        first = min(self.keep_first, len(texts) - last)
        return (
            texts[:first],
            texts[first : len(texts) - last],
            texts[len(texts) - last :],
        )

    def _validate(
        self, originals: List[str], compressed: List[str]
    ) -> tuple[List[str], List[List[str]]]:
        """Keep the original of any entry the compression mangled.

        Fail-closed, per message: one unfaithful entry does not throw away the
        compression of the others, and a rejected entry is recorded so the
        rejection rate is measurable rather than invisible.
        """
        kept: List[str] = []
        rejections: List[List[str]] = []
        for original, text in zip(originals, compressed):
            failures = (
                self.validator.failures(original, text)
                if text and text != original
                else []
            )
            rejections.append(failures)
            if failures:
                self._warn(
                    f'Refined context: rejected a compressed message and kept '
                    f'the original ({"; ".join(failures[:3])}).'
                )
                kept.append('')
            else:
                kept.append(text)
        return kept, rejections

    def _record(
        self,
        head: List[str],
        history: List[str],
        refined: List[str],
        rejected: List[List[str]],
        latest: List[str],
        tail: List[str],
        tail_rejected: List[List[str]],
    ) -> None:
        self.records = [
            MessageRecord(i, 'verbatim-head', len(t), len(t))
            for i, t in enumerate(head)
        ]
        self.records += self._zone_records(
            len(head), history, refined, rejected, 'compressed', 'unchanged'
        )
        structured = self.active_compressor is not None
        self.records += self._zone_records(
            len(head) + len(history),
            latest,
            tail,
            tail_rejected,
            'compressed-structured' if structured else 'verbatim-tail',
            'verbatim-tail',
        )

    @staticmethod
    def _zone_records(
        offset: int,
        originals: List[str],
        refined: List[str],
        rejected: List[List[str]],
        compressed_policy: str,
        untouched_policy: str,
    ) -> List[MessageRecord]:
        records = []
        for j, (original, text) in enumerate(zip(originals, refined)):
            failures = rejected[j] if j < len(rejected) else []
            if failures:
                policy = 'rejected'
            else:
                policy = compressed_policy if text else untouched_policy
            records.append(
                MessageRecord(
                    offset + j,
                    policy,
                    len(original),
                    len(text) if text else len(original),
                    failures,
                )
            )
        return records

    def _skip(
        self,
        head: List[str],
        history: List[str],
        latest: List[str],
        originals: List[str],
        why: str,
    ) -> List[str] | None:
        """Record that nothing was compressed, and hand back Z1 if there is one.

        Skipping compression is not a reason to skip the pinned state block.
        The case where they coincide is the one that most needs the block: a
        node restarted on its very first turn with the tail kept verbatim has
        nothing to compress, and without a resumption marker it reads its own
        pending instruction as a fresh task and starts over.
        """
        self.compressed = False
        self.failure = why
        self._record(
            head,
            history,
            [''] * len(history),
            [[] for _ in history],
            latest,
            [''] * len(latest),
            [[] for _ in latest],
        )
        self._warn(f'Refined context: skipped compression -- {why}.')

        texts = self._with_resumption([''] * len(originals), originals, len(head))
        self._texts = texts if any(texts) else None
        return self._texts

    def _compress(self, history: List[str], latest: List[str]) -> CompressionResult:
        """Compress the history under the configured failure policy."""
        try:
            result = self.compressor.compress(history, question=self._question(latest))
        except Exception as exc:
            self.compressed = False
            self.failure = f'{type(exc).__name__}: {exc}'
            if self.on_error is CompressionFailure.RAISE:
                raise
            self._warn(
                f'Refined context: compression failed ({self.failure}); falling '
                f'back to the uncompressed history. This turn is NOT refined -- '
                f'exclude it when comparing restart modes.'
            )
            return ContextCompressor().compress(history)

        self.compressed = True
        self.failure = None
        return result

    def _question(self, latest: List[str]) -> str:
        """What the node has to act on, for a query-aware compressor only.

        LLMLingua-2 is task-agnostic and discards the question, so building one
        for it would imply a conditioning that does not happen. Compressors
        advertise whether they read it via ``uses_question``.
        """
        if not getattr(self.compressor, 'uses_question', False):
            return ''
        return '\n\n'.join(text for text in latest if text)

    def _warn(self, message: str) -> None:
        if self.logger is None:
            return
        warn = getattr(self.logger, 'warning', None) or self.logger.info
        warn(message)

    def _log(
        self,
        result: CompressionResult,
        head: List[str],
        history: List[str],
        latest: List[str],
        rejected: List[List[str]],
    ) -> None:
        if self.logger is None:
            return
        # Count the messages actually handed to the compressor and accepted.
        # The rest were never sent (empty, attachment-only, tool protocol) or
        # were rejected by the validator, and counting those as compressed
        # misreads the ratio.
        n_rejected = sum(1 for f in rejected if f)
        candidates = sum(1 for text in history if text.strip())
        if self.active_compressor is not None:
            candidates += sum(1 for text in latest if text.strip())
        compressed = candidates - n_rejected
        total = len(head) + len(history) + len(latest)
        tail = (
            f'last {len(latest)} compressed along its structure'
            if self.active_compressor is not None
            else f'last {len(latest)} kept verbatim'
        )
        conditioned = getattr(self.compressor, 'uses_question', False)
        self.logger.info(
            f'Refined context: compressed {compressed} of {total} message(s), '
            f'{result.describe()}; first {len(head)} kept as instruction, '
            f'{tail}'
            + (f', {n_rejected} rejected by the validator' if n_rejected else '')
            + f'; question-conditioned: {conditioned}.'
        )


class RestartWithDiff(Restart):
    def __init__(self, restart_count: int = 1, build_diff_message: Callable | None = None) -> None:
        super().__init__(restart_count)
        self.no_diff_note = "No changes were made to the repository yet."
        self.build_diff_message = build_diff_message

    def set_diff(self, diff_path: Path | None) -> None:
        self.diff_path = diff_path

    def get_context(self) -> Any:
        if not self.context:
            return None
        if not self.adapter:
            raise ValueError('RestartWithDiff needs the node adapter to read the prior context')
        texts = self.adapter.context_as_list(self.context)
        message = self.build_diff_message(self.idx, texts, self.diff_path)
        return self.adapter.context_from_list([message], self.context)

    def restore_input(self, adapter: 'NodeAdapter', injected_context: Any) -> None:
        """Splice the one-shot engineered message back out after the retry.

        ``get_context()`` collapses the node's whole history into a single
        message meant only to steer the one retried call this triggers --
        not to become the node's history from here on. Left in place, it
        sits at the head of an ever-growing, never-pruned transcript
        (``context_window: -1``) and gets replayed verbatim on every future
        turn, so a Coder restarted once keeps being told to inspect an
        overlay diff long after that restart is over. This restores the
        real pre-fault history (``self.context``, snapshotted before the
        fault ran) and keeps only what this execution actually appended
        after the synthetic message.
        """
        if not self.context:
            return
        injected_len = len(injected_context) if injected_context else 0
        appended = list(adapter.input or [])[injected_len:]
        adapter.set_input(list(self.context) + appended)

    def _merge_history(self, texts):
        pass


class RestartWithDiffSections(Restart):
    def __init__(self, restart_count: int = 1, build_diff_message: Callable | None = None, keep_labels: List[str] | None = None) -> None:
        super().__init__(restart_count)
        self.no_diff_note = "No changes were made to the repository yet."
        self.build_diff_message = build_diff_message
        self.keep_labels = keep_labels
        self._pattern = build_label_pattern(self.keep_labels)

    def set_diff(self, diff_path: Path | None) -> None:
        self.diff_path = diff_path

    def get_context(self) -> Any:
        if not self.context:
            return None
        if not self.adapter:
            raise ValueError('RestartWithDiff needs the node adapter to read the prior context')
        texts = self.adapter.context_as_list(self.context)
        filtered = [keep_labelled_sections(t, self._pattern) for t in texts]
        message = self.build_diff_message(self.idx, filtered, self.diff_path)
        return self.adapter.context_from_list([message], self.context)

    def restore_input(self, adapter: 'NodeAdapter', injected_context: Any) -> None:
        """Splice the one-shot engineered message back out after the retry.

        Same reasoning as ``RestartWithDiff.restore_input`` -- the message
        ``get_context()`` returns is a one-time nudge for the retried call,
        not a replacement for the node's real history. Without this, it sits
        at the head of an ever-growing, never-pruned transcript
        (``context_window: -1``) and gets replayed on every future turn.
        """
        if not self.context:
            return
        injected_len = len(injected_context) if injected_context else 0
        appended = list(adapter.input or [])[injected_len:]
        adapter.set_input(list(self.context) + appended)