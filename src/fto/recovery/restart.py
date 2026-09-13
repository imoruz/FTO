from copy import deepcopy
from typing import TYPE_CHECKING, Any, List

from dataclasses import dataclass, field

from fto.recovery.compression import (
    CompressionFailure,
    CompressionResult,
    CompressionValidator,
    ContextCompressor,
    LLMLinguaCompressor,
)

if TYPE_CHECKING:
    from fto.adapters.node.node import NodeAdapter


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

    def get_context(self) -> Any:
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
        self._texts: List[str] | None = None

    def set_context(self, context: Any, adapter: 'NodeAdapter | None' = None) -> None:
        super().set_context(context, adapter=adapter)
        self._texts = None
        self.last_result = None
        self.compressed = None
        self.failure = None
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

    def _refine(self) -> List[str] | None:
        """The refined text of every message, or None if there is nothing to do.

        Compression runs once per snapshot: further attempts on the same node
        replay the same refined context instead of paying for the model again.
        """
        if self._texts is not None:
            return self._texts

        texts = self.adapter.context_as_list(self.context)
        head, history, latest = self._split(texts)
        if not any(text.strip() for text in history):
            self._skip(
                head,
                history,
                latest,
                'nothing to compress between the '
                'protected head and the kept-verbatim tail',
            )
            return None
        if sum(len(text) for text in history) < self.min_chars:
            self._skip(
                head,
                history,
                latest,
                f'history under min_chars '
                f'({self.min_chars}); not worth the semantic risk',
            )
            return None

        result = self._compress(history, latest)
        active = self._compress_active(latest)
        combined = _merge(result, active)
        if self.min_saving and combined.rate > 1 - self.min_saving:
            self._skip(
                head,
                history,
                latest,
                f'saving {1 - combined.rate:.1%} '
                f'below min_saving ({self.min_saving:.0%})',
            )
            self.last_result = combined
            return None

        refined, rejected = self._validate(history, result.texts)
        if active is not None:
            tail, tail_rejected = self._validate(latest, active.texts)
        else:
            tail, tail_rejected = [''] * len(latest), [[] for _ in latest]
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
        self._texts = [''] * len(head) + refined + tail
        return self._texts

    def _compress_active(self, latest: List[str]) -> CompressionResult | None:
        """Compress the newest message(s), if a compressor was given for them.

        None means the tail is kept verbatim, which is the default: it is what
        the node has to act on. A ``StructuredCompressor`` here shrinks it
        along its own labels instead, so the scaffold the next agent reads
        stays intact and each field is compressed by what it is worth.
        """
        if self.active_compressor is None or not any(t.strip() for t in latest):
            return None
        return self.active_compressor.compress(latest)

    def _split(self, texts: List[str]) -> tuple[List[str], List[str], List[str]]:
        """Protected head, compressible history, kept-verbatim tail."""
        first = min(self.keep_first, len(texts))
        last = min(self.keep_last, len(texts) - first)
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
        self, head: List[str], history: List[str], latest: List[str], why: str
    ) -> None:
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
