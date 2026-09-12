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

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List

# LLMLingua-2: a BERT-size token classifier, task-agnostic and cheap to run.
LLMLINGUA2_MODEL = 'microsoft/llmlingua-2-xlm-roberta-large-meetingbank'
# (Long)LLMLingua: perplexity-based, needs a causal LM to score tokens.
LONGLLMLINGUA_MODEL = 'NousResearch/Llama-2-7b-hf'

# Structural characters worth keeping so compressed history stays readable.
# '.' is deliberately absent: forcing it makes LLMLingua-2 re-join it as its
# own word, which turns "6.4.6" into "6. 4. 6" and "cpe.go" into "cpe. go".
STRUCTURAL_FORCE_TOKENS = ['\n', ':', '?', '!']

# Words whose loss inverts a requirement. Token pruning drops function words
# by design, which turns "must never emit fortios" into "emit fortios" and
# "No changes to tests" into "changes tests" -- a compressed instruction that
# says the opposite of the original is worse than one that says less, so these
# are pinned. Matching is case-sensitive, hence the sentence-initial variants.
NEGATION_FORCE_TOKENS = [
    'not',
    'Not',
    'no',
    'No',
    'none',
    'None',
    'never',
    'Never',
    'only',
    'Only',
    'must',
    'Must',
    'cannot',
    'Cannot',
    'without',
    'except',
    'unless',
]

DEFAULT_FORCE_TOKENS = STRUCTURAL_FORCE_TOKENS + NEGATION_FORCE_TOKENS

# Spans token pruning must not touch: fenced code blocks and inline backtick
# spans. Pruned code is not merely shorter, it is wrong -- `cpe.go` comes back
# as `cpe.`, "cpe:2.3:h:fortinet:%s" as ":2.:fortinet:", "6.4.6" as "6. 4. 6"
# -- leaving the node a path or literal that looks real and is not. Unlike
# force_tokens, this protection applies in both LLMLingua modes.
PROTECTED_SPANS = re.compile(r'```[\s\S]*?```|`[^`\n]+`')


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


class CompressionFailure(StrEnum):
    """What a restart does when compression raises.

    There is no safe default guess: silently falling back to uncompressed text
    makes a refined restart indistinguishable from an all-context one, and a
    benchmark that mixes the two without saying so is unreadable. So the
    choice is explicit, and either way the restart records whether the turn
    was actually compressed.
    """

    #: Let the error out and fail the run. Nothing is recorded as refined
    #: that was not refined.
    RAISE = 'raise'
    #: Fall back to the no-op passthrough, log it as a warning, and mark the
    #: turn uncompressed so it can be filtered out of the results.
    PASSTHROUGH = 'passthrough'


class ContextCompressor:
    """Shrinks context entries, one compressed entry per input entry.

    The base class is a no-op passthrough, which makes it a usable stand-in
    whenever compression should be disabled without changing the restart mode.
    """

    #: Whether ``compress`` actually reads ``question``. Callers check this
    #: instead of passing a question that will be silently discarded.
    uses_question: bool = False

    def compress(self, contexts: List[str], question: str = '') -> CompressionResult:
        """Compress ``contexts``, optionally conditioned on ``question``.

        ``question`` is what the node is about to act on -- the messages the
        restart keeps verbatim. Only a query-aware compressor uses it, and
        only those set ``uses_question``; for everything else it is ignored,
        so do not read a conditioning guarantee into the signature.
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


def _reassemble(plan: List[List]) -> str:
    """Put an entry's pieces back together.

    The compressor strips the whitespace around a fragment it was given, so a
    single space goes back wherever that would otherwise weld a compressed
    word onto the code span next to it.
    """
    out: List[str] = []
    for _, fragment in plan:
        if not fragment:
            continue
        if out and not out[-1][-1:].isspace() and not fragment[:1].isspace():
            out.append(' ')
        out.append(fragment)
    return ''.join(out)


@dataclass
class LLMLinguaCompressor(ContextCompressor):
    """Token pruning via LLMLingua (Microsoft Research).

    Two modes, both keeping the output aligned with the input:

    ``use_llmlingua2=True`` (default) compresses every entry in a single
    batched call and reads back ``compressed_prompt_list``. Context-level
    filtering is switched off because it drops whole entries, which would break
    that alignment. This mode honours ``force_tokens``.

    LLMLingua-2 is **task-agnostic and ignores ``question``** -- upstream,
    ``compress_prompt`` forwards neither ``question`` nor ``instruction`` to
    ``compress_prompt_llmlingua2``, and the paper frames the method as uniform
    relevance scoring with no awareness of what the node is about to do. So
    ``uses_question`` is False in this mode and the restart layer does not
    build a question for it. Do not "fix" that by passing one: it would imply
    a conditioning that is not happening. What keeps the history faithful here
    is ``force_tokens``, ``PROTECTED_SPANS`` and ``min_fragment_chars``, not
    relevance to the active message -- so the rate needs needle-testing rather
    than trusting the scorer to know what matters (see tests/recovery/
    test_needle.py; on Planner/Coder plans rate 0.4 held every needle and
    0.25 began shortening line ranges).

    ``use_llmlingua2=False`` runs LongLLMLingua once per entry, conditioned on
    ``question``: the v1 API only ever returns one flat string, so per-entry
    calls are what keeps entries separable. This is the only mode where
    ``uses_question`` is True. ``force_tokens`` is not supported upstream
    here and is ignored.

    With ``protect_code`` (the default) only prose reaches the model: fenced
    code blocks and inline backtick spans are cut out, held aside, and put
    back where they were. Each entry therefore becomes several fragments, so
    the model scores each run of prose without the code around it.

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
    # Keep fenced code blocks and backtick spans out of the compressor.
    # Turn it off only to measure what protection is worth.
    protect_code: bool = True
    # Prose runs shorter than this are kept verbatim. Protecting code splits a
    # message into many short fragments -- in a plan dense with backticks most
    # of them are a few words of glue -- and each one costs its own padded
    # forward pass while saving almost nothing. 0 compresses every fragment.
    min_fragment_chars: int = 80
    # Escape hatch for anything else ``compress_prompt`` accepts.
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_name is None:
            self.model_name = (
                LLMLINGUA2_MODEL if self.use_llmlingua2 else LONGLLMLINGUA_MODEL
            )

    @property
    def uses_question(self) -> bool:
        """Only LongLLMLingua conditions on the question; LLMLingua-2 ignores it."""
        return not self.use_llmlingua2

    def compress(self, contexts: List[str], question: str = '') -> CompressionResult:
        entries = [text if isinstance(text, str) else '' for text in contexts]
        plans = [self._plan(text) for text in entries]

        # Every prose fragment worth compressing, across every entry, in
        # order. Entries with nothing to compress (empty, attachment-only,
        # tool protocol, or nothing but code) contribute none and are never
        # shown to the model.
        todo = [
            (i, j)
            for i, plan in enumerate(plans)
            for j, (compressible, fragment) in enumerate(plan)
            if compressible and len(fragment.strip()) >= max(1, self.min_fragment_chars)
        ]
        if not todo:
            return CompressionResult(entries)

        payload = [plans[i][j][1] for i, j in todo]
        compressed = (
            self._compress_batch(payload)
            if self.use_llmlingua2
            else self._compress_each(payload, question)
        )
        for (i, j), text in zip(todo, compressed):
            plans[i][j][1] = text

        texts = [_reassemble(plan) for plan in plans]
        return CompressionResult(
            texts,
            # Counted over whole entries, protected spans included, so the
            # ratio reports the reduction the node actually sees.
            origin_tokens=self._count(entries),
            compressed_tokens=self._count(texts),
        )

    def _plan(self, text: str) -> List[List]:
        """Cut one entry into alternating [compressible, fragment] pieces."""
        if not text:
            return []
        if not self.protect_code:
            return [[True, text]]

        pieces: List[List] = []
        at = 0
        for span in PROTECTED_SPANS.finditer(text):
            if span.start() > at:
                pieces.append([True, text[at : span.start()]])
            pieces.append([False, span.group()])
            at = span.end()
        if at < len(text):
            pieces.append([True, text[at:]])
        return pieces

    def _compress_batch(self, contexts: List[str]) -> List[str]:
        # No question here on purpose: compress_prompt drops it before
        # reaching compress_prompt_llmlingua2 (see the class docstring).
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
