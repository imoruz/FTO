from copy import deepcopy
from typing import TYPE_CHECKING, Any, List

from fto.recovery.compression import (
    CompressionResult,
    ContextCompressor,
    LLMLinguaCompressor,
)

if TYPE_CHECKING:
    from fto.adapters.node.node import NodeAdapter


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
    ) -> None:
        super().__init__(restart_count)
        self.compressor = (
            compressor if compressor is not None else LLMLinguaCompressor()
        )
        # At least one message has to survive intact, or the node is restarted
        # with no statement of what it is being asked to do.
        self.keep_last = max(1, keep_last)
        self.logger = logger
        self.last_result: CompressionResult | None = None
        self._texts: List[str] | None = None

    def set_context(self, context: Any, adapter: 'NodeAdapter | None' = None) -> None:
        super().set_context(context, adapter=adapter)
        self._texts = None
        self.last_result = None

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
        if len(texts) <= self.keep_last:
            # Nothing here but messages that are kept verbatim anyway.
            return None

        history, latest = self._split(texts)
        result = self.compressor.compress(history, question=self._question(latest))
        self.last_result = result
        self._log(result, kept=len(latest))

        # '' means "leave that message as it was", which is exactly what the
        # kept-verbatim tail needs -- and the right thing to do for a history
        # entry that compressed down to nothing.
        self._texts = list(result.texts) + [''] * len(latest)
        return self._texts

    def _split(self, texts: List[str]) -> tuple[List[str], List[str]]:
        split = len(texts) - self.keep_last
        return texts[:split], texts[split:]

    def _question(self, latest: List[str]) -> str:
        """What the node has to act on, for query-aware compressors."""
        return '\n\n'.join(text for text in latest if text)

    def _log(self, result: CompressionResult, kept: int) -> None:
        if self.logger is None:
            return
        compressed = len(result.texts)
        self.logger.info(
            f'Refined context: compressed {compressed} of '
            f'{compressed + kept} message(s), {result.describe()}; '
            f'last {kept} message(s) kept verbatim.'
        )
