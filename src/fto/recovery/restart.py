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
        if self.min_saving and result.rate > 1 - self.min_saving:
            self._skip(
                head,
                history,
                latest,
                f'saving {1 - result.rate:.1%} '
                f'below min_saving ({self.min_saving:.0%})',
            )
            self.last_result = result
            return None

        refined = self._validate(history, result.texts)
        self.last_result = result
        self._record(head, history, refined, latest)
        self._log(result, head=head, history=history, latest=latest)

        # '' means "leave that message as it was", which is exactly what the
        # protected head and the kept-verbatim tail need -- and the right thing
        # to do for a history entry that compressed down to nothing.
        self._texts = [''] * len(head) + refined + [''] * len(latest)
        return self._texts

    def _split(self, texts: List[str]) -> tuple[List[str], List[str], List[str]]:
        """Protected head, compressible history, kept-verbatim tail."""
        first = min(self.keep_first, len(texts))
        last = min(self.keep_last, len(texts) - first)
        return (
            texts[:first],
            texts[first : len(texts) - last],
            texts[len(texts) - last :],
        )

    def _validate(self, history: List[str], compressed: List[str]) -> List[str]:
        """Keep the original of any entry the compression mangled.

        Fail-closed, per message: one unfaithful entry does not throw away the
        compression of the others, and a rejected entry is recorded so the
        rejection rate is measurable rather than invisible.
        """
        kept: List[str] = []
        self._rejections: List[List[str]] = []
        for original, text in zip(history, compressed):
            failures = (
                self.validator.failures(original, text)
                if text and text != original
                else []
            )
            self._rejections.append(failures)
            if failures:
                self._warn(
                    f'Refined context: rejected a compressed message and kept '
                    f'the original ({"; ".join(failures[:3])}).'
                )
                kept.append('')
            else:
                kept.append(text)
        return kept

    def _record(
        self,
        head: List[str],
        history: List[str],
        refined: List[str],
        latest: List[str],
    ) -> None:
        rejections = getattr(self, '_rejections', [[]] * len(history))
        self.records = [
            MessageRecord(i, 'verbatim-head', len(t), len(t))
            for i, t in enumerate(head)
        ]
        for j, (original, text) in enumerate(zip(history, refined)):
            failures = rejections[j]
            policy = 'rejected' if failures else 'compressed' if text else 'unchanged'
            self.records.append(
                MessageRecord(
                    len(head) + j,
                    policy,
                    len(original),
                    len(text) if text else len(original),
                    failures,
                )
            )
        offset = len(head) + len(history)
        self.records += [
            MessageRecord(offset + i, 'verbatim-tail', len(t), len(t))
            for i, t in enumerate(latest)
        ]

    def _skip(
        self, head: List[str], history: List[str], latest: List[str], why: str
    ) -> None:
        self.compressed = False
        self.failure = why
        self._record(head, history, [''] * len(history), latest)
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
    ) -> None:
        if self.logger is None:
            return
        # Count the messages actually handed to the compressor and accepted.
        # The rest were never sent (empty, attachment-only, tool protocol) or
        # were rejected by the validator, and counting those as compressed
        # misreads the ratio.
        rejected = sum(1 for f in getattr(self, '_rejections', []) if f)
        compressed = sum(1 for text in history if text.strip()) - rejected
        total = len(head) + len(history) + len(latest)
        conditioned = getattr(self.compressor, 'uses_question', False)
        self.logger.info(
            f'Refined context: compressed {compressed} of {total} message(s), '
            f'{result.describe()}; first {len(head)} kept as instruction, '
            f'last {len(latest)} kept verbatim'
            + (f', {rejected} rejected by the validator' if rejected else '')
            + f'; question-conditioned: {conditioned}.'
        )
