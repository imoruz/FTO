"""Context compression used by restart with refined context.

Everything here works on plain text: a list of entries, one per message the
restarting node had queued. The node adapters own the translation between a
MAS-native message list and that text view (``NodeAdapter.context_as_list`` /
``NodeAdapter.context_from_list``), so nothing in this module needs to know
which framework produced the messages.

Compressors must keep the returned list aligned with the input list, entry for
entry -- the caller maps each compressed entry back onto the message it came
from, which is what lets roles, sources and attachments survive compression.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

# LLMLingua-2: a BERT-size token classifier, task-agnostic and cheap to run.
LLMLINGUA2_MODEL = 'microsoft/llmlingua-2-xlm-roberta-large-meetingbank'
# (Long)LLMLingua: perplexity-based, needs a causal LM to score tokens.
LONGLLMLINGUA_MODEL = 'NousResearch/Llama-2-7b-hf'

# Structural characters worth keeping so compressed history stays readable.
DEFAULT_FORCE_TOKENS = ['\n', '.', ':', '?', '!']


@dataclass
class CompressionResult:
    """Compressed entries plus what the compression cost in tokens."""

    texts: List[str]
    origin_tokens: int = 0
    compressed_tokens: int = 0

    @property
    def rate(self) -> float:
        """Compressed size over original size; 1.0 when nothing was compressed."""
        if not self.origin_tokens:
            return 1.0
        return self.compressed_tokens / self.origin_tokens

    def describe(self) -> str:
        return (
            f'{self.origin_tokens} -> {self.compressed_tokens} tokens '
            f'({self.rate * 100:.1f}% of original)'
        )


class ContextCompressor:
    """Shrinks context entries, one compressed entry per input entry.

    The base class is a no-op passthrough, which makes it a usable stand-in
    whenever compression should be disabled without changing the restart mode.
    """

    def compress(self, contexts: List[str], question: str = '') -> CompressionResult:
        """Compress ``contexts``, optionally conditioned on ``question``.

        ``question`` is what the node is about to act on (the messages the
        restart keeps verbatim). Compressors that support query-aware
        compression use it to decide what in the history still matters.
        """
        texts = list(contexts)
        return CompressionResult(texts)


_MODEL_CACHE: Dict[Any, Any] = {}


def _load(model_name: str, use_llmlingua2: bool, device_map: str | None) -> Any:
    """Build a ``PromptCompressor``, once per (model, mode, device).

    Restarts happen repeatedly inside one run and the scoring model is hundreds
    of megabytes, so it is loaded lazily on first use and then reused. The
    llmlingua import lives here too: it pulls in torch and transformers, which
    no other part of fto needs.
    """
    key = (model_name, use_llmlingua2, device_map)
    compressor = _MODEL_CACHE.get(key)
    if compressor is not None:
        return compressor

    from llmlingua import PromptCompressor

    compressor = PromptCompressor(
        model_name=model_name,
        use_llmlingua2=use_llmlingua2,
        device_map=device_map or _default_device(),
    )
    _MODEL_CACHE[key] = compressor
    return compressor


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class LLMLinguaCompressor(ContextCompressor):
    """Token pruning via LLMLingua (Microsoft Research).

    Two modes, both keeping the output aligned with the input:

    ``use_llmlingua2=True`` (default) compresses every entry in a single
    batched call and reads back ``compressed_prompt_list``. Context-level
    filtering is switched off because it drops whole entries, which would break
    that alignment. This mode honours ``force_tokens``.

    ``use_llmlingua2=False`` runs LongLLMLingua once per entry, conditioned on
    ``question``: the v1 API only ever returns one flat string, so per-entry
    calls are what keeps entries separable. ``force_tokens`` is not supported
    upstream in this mode and is ignored.

    Failures are deliberately not swallowed: a silent fallback to uncompressed
    text would make a refined restart indistinguishable from an all-context
    one.
    """

    model_name: str | None = None
    use_llmlingua2: bool = True
    device_map: str | None = None
    # Fraction of the original tokens to keep (0.55 -> drop about 45%).
    rate: float = 0.55
    # Hard token budget; -1 leaves ``rate`` in charge.
    target_token: int = -1
    # Strings compression must never drop. MAS-specific markers belong here:
    # control keywords the workflow routes on, report field labels, and so on.
    force_tokens: List[str] = field(default_factory=lambda: list(DEFAULT_FORCE_TOKENS))
    force_reserve_digit: bool = True
    drop_consecutive: bool = False
    # Escape hatch for anything else ``compress_prompt`` accepts.
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_name is None:
            self.model_name = (
                LLMLINGUA2_MODEL if self.use_llmlingua2 else LONGLLMLINGUA_MODEL
            )

    def compress(self, contexts: List[str], question: str = '') -> CompressionResult:
        entries = [text if isinstance(text, str) else '' for text in contexts]
        # Entries with nothing to compress (empty, attachment-only, tool
        # protocol) stay put and are never handed to the model.
        indices = [i for i, text in enumerate(entries) if text.strip()]
        if not indices:
            return CompressionResult(entries)

        payload = [entries[i] for i in indices]
        compressed = (
            self._compress_batch(payload)
            if self.use_llmlingua2
            else self._compress_each(payload, question)
        )

        texts = list(entries)
        for i, text in zip(indices, compressed):
            texts[i] = text
        return CompressionResult(
            texts,
            origin_tokens=self._count(payload),
            compressed_tokens=self._count(compressed),
        )

    def _compress_batch(self, contexts: List[str]) -> List[str]:
        result = self._compressor().compress_prompt(
            contexts,
            rate=self.rate,
            target_token=self.target_token,
            # Would drop whole entries and desynchronise the result list.
            use_context_level_filter=False,
            force_tokens=list(self.force_tokens),
            force_reserve_digit=self.force_reserve_digit,
            drop_consecutive=self.drop_consecutive,
            **self.params,
        )
        texts = result.get('compressed_prompt_list')
        if texts is None or len(texts) != len(contexts):
            raise RuntimeError(
                f'{self.model_name} returned '
                f'{0 if texts is None else len(texts)} compressed entries for '
                f'{len(contexts)} contexts; cannot map them back onto messages.'
            )
        return list(texts)

    def _compress_each(self, contexts: List[str], question: str) -> List[str]:
        compressor = self._compressor()
        # LongLLMLingua ranks context against a question; with nothing to
        # condition on it degrades to plain LLMLingua.
        conditioned = bool(question and question.strip())
        options = dict(
            rate=self.rate,
            target_token=self.target_token,
            rank_method='longllmlingua' if conditioned else 'llmlingua',
            condition_in_question='after_condition' if conditioned else 'none',
            condition_compare=conditioned,
            reorder_context='sort',
            dynamic_context_compression_ratio=0.3,
            context_budget='+100',
            # The question is a message of its own, kept verbatim by the
            # caller; appending it to every entry would duplicate it.
            concate_question=False,
        )
        options.update(self.params)

        texts = []
        for context in contexts:
            result = compressor.compress_prompt(
                [context], question=question if conditioned else '', **options
            )
            texts.append(result['compressed_prompt'])
        return texts

    def _count(self, texts: List[str]) -> int:
        """Token count as LLMLingua itself reports it, for both modes alike."""
        tokenizer = self._compressor().oai_tokenizer
        return sum(len(tokenizer.encode(text)) for text in texts if text)

    def _compressor(self) -> Any:
        return _load(self.model_name, self.use_llmlingua2, self.device_map)
