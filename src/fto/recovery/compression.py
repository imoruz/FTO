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
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Dict, List

# LLMLingua-2: a BERT-size token classifier, task-agnostic and cheap to run.
LLMLINGUA2_MODEL = 'microsoft/llmlingua-2-xlm-roberta-large-meetingbank'
# (Long)LLMLingua: perplexity-based, needs a causal LM to score tokens.
LONGLLMLINGUA_MODEL = 'NousResearch/Llama-2-7b-hf'

# Structural characters worth keeping so compressed history stays readable.
#
# The markdown hashes are here because a plan's own scaffold ("## Fix Plan",
# "### File path:", "#### Change N:") is the addressing scheme the receiving
# agent reports against; without them the compressor dissolves the headings
# and the enumeration goes with them. They are a backstop only -- heading
# *lines* are held out of the compressor wholesale by HEADING_SPANS, because
# pinning "####" alone still yields "### # 1:" (measured).
#
# '.' is deliberately absent, and this is a considered deviation from the
# restart spec's shared token guard: forcing it makes LLMLingua-2 re-join it
# as its own word, which turns "6.4.6" into "6. 4. 6" and "cpe.go" into
# "cpe. go". Re-measured against the real model on a Planner fix plan before
# writing this down -- see docs/compression.md.
STRUCTURAL_FORCE_TOKENS = ['\n', ':', '?', '!', '#', '##', '###', '####', '-', '`']

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

# Extensions the benchmark's instances actually contain. The spec's own list
# is Python-flavoured; SWE-bench-Pro also ships Go, JS/TS and Java repos, and
# a path is only protected if its extension is recognised.
SOURCE_EXTENSIONS = (
    'py|pyi|yaml|yml|json|txt|md|rst|toml|cfg|ini|sh|sql|'
    'go|mod|js|jsx|ts|tsx|vue|rs|java|kt|rb|php|c|h|cc|cpp|hpp|cs|swift'
)

# A file path written as bare prose. Protection is by backtick everywhere
# else in this module, but agents routinely name paths unquoted -- and a
# pruned path is the worst possible output, because the node calls the file
# tool with something that looks real and does not exist. Measured at rate
# 0.33: "lib/ansible/utils/unsafe_proxy.py" came back as "_proxy.py".
PATH_SPANS = re.compile(rf'(?<![\w/.-])[\w.-]*(?:/[\w.-]+)*\.(?:{SOURCE_EXTENSIONS})\b')

# "line 61", "lines 105-113", "L412". The whole review loop is
# read_file_segment against exact line numbers, and fine-grained pruning
# shortens a range to its first half ("lines 87-104" -> "87") even with
# force_reserve_digit on, because the end of the range is a separate token.
LINE_REF_SPANS = re.compile(r'\b(?:lines?|L)\s*\d+(?:\s*[-–—]\s*\d+)?', re.IGNORECASE)

# A markdown heading, whole line. "#### Change N:" is the enumeration the
# receiving agent reports against and "### <path>" is the addressing scheme;
# both are headers the compressor must not touch. Pinning the hashes in
# force_tokens is not enough on its own -- measured, "#### Change 1: widen
# the wrapper set" came back as "### # 1: widen wrapper set".
HEADING_SPANS = re.compile(r'^[ \t]*#{1,6}[ \t]+[^\n]*', re.MULTILINE)


# Words whose loss flips a requirement, for checking a compression after the
# fact. Wider than NEGATION_FORCE_TOKENS on purpose: force_tokens pin single
# tokens, but "DON'T" is an apostrophe contraction the pinning cannot express,
# so the validator is what catches it.
NEGATION_WORDS = re.compile(
    r'\b(?:not|no|none|never|neither|nor|without|except|unless|cannot'
    r"|can't|don't|doesn't|didn't|won't|isn't|aren't|wasn't|shouldn't"
    r"|mustn't|nil|null)\b",
    re.IGNORECASE,
)

# An XML-ish delimiter, the kind agent prompts wrap their sections in
# (<pr_description>, </uploaded_files>).
DELIMITERS = re.compile(r'</?[A-Za-z_][\w.-]*>')


# Symbols an agent has already named: a dotted attribute chain, a CamelCase
# type, a snake_case or dunder function. Harvested from the message itself and
# pinned for that call only, so the identifiers the plan is *about* survive
# even where they were written as bare prose rather than in backticks. The
# shapes are deliberately restricted to names that cannot occur as a fragment
# of an ordinary English word: LLMLingua-2 implements a multi-token force
# token as a plain substring replace, so pinning a bare lowercase word would
# rewrite the middle of unrelated text.
IDENTIFIER_SHAPES = re.compile(
    r'\b[A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)+\b'  # pkg.module.attr
    r'|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b'  # CamelCase
    r'|\b\w*[a-z0-9]_\w+\b'  # snake_case
    r'|\b__\w+__\b'  # __all__
)

#: Hard cap on how many harvested identifiers are pinned. Each force token
#: costs a pass over the text inside the library, so an unbounded allowlist
#: turns a cheap guard into the dominant cost on a long plan.
MAX_HARVESTED_IDENTIFIERS = 64

#: llmlingua allocates exactly this many `[NEWi]` placeholder tokens when the
#: model loads (see PromptCompressor.init_llmlingua2) and asserts
#: len(force_tokens) <= this at compress time. Not configurable per call, so
#: anything that adds to force_tokens has to share this budget with whatever
#: the caller already pinned.
LLMLINGUA_MAX_FORCE_TOKENS = 100


def harvested_identifier_budget(force_tokens: List[str]) -> int:
    """How many identifiers ``harvest_identifiers`` may add on top of ``force_tokens``.

    Keeps ``len(force_tokens) + len(pinned) <= LLMLINGUA_MAX_FORCE_TOKENS`` no
    matter how many fixed force tokens a given MAS config already carries,
    instead of relying on ``MAX_HARVESTED_IDENTIFIERS`` alone staying small
    enough for whatever force_tokens list happens to be active.
    """
    return max(0, min(MAX_HARVESTED_IDENTIFIERS, LLMLINGUA_MAX_FORCE_TOKENS - len(force_tokens)))


def harvest_identifiers(
    texts: List[str], limit: int = MAX_HARVESTED_IDENTIFIERS
) -> List[str]:
    """The code symbols named in ``texts``, to pin for this call only.

    LLMLingua-2 scores a bare identifier like any other word, so a name the
    plan is built around ("wrap_var", "AnsibleUnsafeBytes", "__all__") is
    dropped as readily as a conjunction unless something holds it. Backticked
    names are already held out by ``PROTECTED_SPANS``; this covers the ones
    the agent wrote unquoted, which in practice is most of them.

    Longest first, because the library replaces force tokens in order and a
    short name that is a prefix of a longer one would otherwise claim it.
    """
    seen: Dict[str, None] = {}
    for text in texts:
        if not text:
            continue
        for match in IDENTIFIER_SHAPES.finditer(text):
            name = match.group()
            if len(name) >= 3:
                seen[name] = None
    return sorted(seen, key=len, reverse=True)[:limit]


def unweld(text: str, tokens: List[str]) -> str:
    """Re-space a pinned token the compressor joined onto the previous word.

    LLMLingua-2 honours a multi-token force token by swapping it for an
    internal placeholder and mapping it back afterwards. When the word before
    it is pruned the placeholder ends up welded to whatever now precedes it --
    "Called" + "describe_available_files" comes back as
    "Calleddescribe_available_files". The identifier is intact and the meaning
    survives, but the seam reads as a different symbol than the one that is
    there, and the next agent greps for symbols.

    Only ever inserts a space, and only where an alphanumeric character butts
    directly against a pinned name. Tokens are applied longest first so a name
    that is a suffix of a longer one does not claim it.
    """
    for token in sorted(tokens, key=len, reverse=True):
        if not token or not token[:1].isidentifier() and not token.startswith('_'):
            continue
        text = re.sub(rf'(?<=[A-Za-z0-9]){re.escape(token)}', f' {token}', text)
    return text


def guard_control_literals(text: str, literals: List[str], sentinel: str) -> str:
    """Stop a compressed history block from ending in a routing keyword.

    The workflow's edges match on literal output -- here, a message ending in
    ``TASK_COMPLETE`` routes to the exit node. A history entry that compresses
    down to something ending in that keyword is a live hazard: it satisfies a
    terminal regex without any agent having decided anything. Appending a
    sentinel line costs nothing and removes the class of failure.
    """
    if not literals:
        return text
    stripped = text.rstrip()
    if any(stripped.endswith(literal) for literal in literals):
        return stripped + '\n' + sentinel
    return text


@dataclass
class CompressionValidator:
    """Checks a compressed entry is a faithful shortening of the original.

    Token pruning has no notion of meaning, so it will happily produce text
    that reads as an instruction and says the opposite of one. These are the
    losses that are objectively detectable by comparing the two strings, so
    they are checked at runtime rather than hoped about: the caller rejects a
    failing entry and keeps the original.
    """

    #: Every negation in the original must still be there. "DON'T have to
    #: modify the testing logic" compressing to "'T modify testing logic" is
    #: the case this exists for.
    negations: bool = True
    #: Backticked spans and fenced blocks must survive verbatim. Guaranteed by
    #: PROTECTED_SPANS, so this is a cheap regression check on that guarantee.
    code_spans: bool = True
    #: A section delimiter present in the original must still be present, whole.
    delimiters: bool = True
    #: Bare file paths and "line N" / "lines N-M" references must survive
    #: verbatim. Guaranteed by PATH_SPANS and LINE_REF_SPANS, so like
    #: code_spans this is a regression check on that guarantee -- and the one
    #: that catches a halved line range, which is otherwise invisible because
    #: the truncated version still reads as a valid reference.
    identifiers: bool = True

    def failures(self, original: str, compressed: str) -> List[str]:
        """Every reason this compression should be rejected; empty means fine."""
        reasons: List[str] = []
        if self.negations:
            reasons += self._missing_negations(original, compressed)
        if self.code_spans:
            reasons += self._missing_code_spans(original, compressed)
        if self.identifiers:
            reasons += self._missing_identifiers(original, compressed)
        if self.delimiters:
            reasons += self._missing_delimiters(original, compressed)
        return reasons

    @staticmethod
    def _missing_negations(original: str, compressed: str) -> List[str]:
        from collections import Counter

        before = Counter(w.lower() for w in NEGATION_WORDS.findall(original))
        after = Counter(w.lower() for w in NEGATION_WORDS.findall(compressed))
        return [
            f'negation {word!r} lost ({count}x -> {after[word]}x)'
            for word, count in before.items()
            if after[word] < count
        ]

    @staticmethod
    def _missing_code_spans(original: str, compressed: str) -> List[str]:
        lost = [
            span.group()
            for span in PROTECTED_SPANS.finditer(original)
            if span.group() not in compressed
        ]
        return [f'code span {span!r} lost' for span in dict.fromkeys(lost)]

    @staticmethod
    def _missing_identifiers(original: str, compressed: str) -> List[str]:
        lost = [
            span.group()
            for pattern in (PATH_SPANS, LINE_REF_SPANS)
            for span in pattern.finditer(original)
            if span.group() not in compressed
        ]
        return [f'reference {span!r} lost' for span in dict.fromkeys(lost)]

    @staticmethod
    def _missing_delimiters(original: str, compressed: str) -> List[str]:
        lost = [
            tag.group()
            for tag in DELIMITERS.finditer(original)
            if tag.group() not in compressed
        ]
        reasons = [f'delimiter {tag!r} lost' for tag in dict.fromkeys(lost)]
        if original.count('```') % 2 == 0 and compressed.count('```') % 2:
            reasons.append('fenced block left unbalanced')
        return reasons


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
_PROTECTED_CACHE: Dict[Any, Any] = {}


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
    # Collapse a run of the same forced token. Without it the pinned
    # structural characters pile up in the output as "\n\n\n:::" once the
    # words between them are pruned.
    drop_consecutive: bool = True
    # Keep fenced code blocks and backtick spans out of the compressor.
    # Turn it off only to measure what protection is worth.
    protect_code: bool = True
    # Keep bare file paths and "line N" / "lines N-M" references out of the
    # compressor too. Protection by backtick only covers what the agent
    # happened to quote, and an unquoted path or a halved line range is worse
    # than a dropped sentence: the node acts on it and is wrong.
    protect_identifiers: bool = True
    # Keep whole markdown heading lines out of the compressor. A plan's
    # headings are its addressing scheme, not prose.
    protect_headings: bool = True
    # Symbols to pin for this call. None means harvest them from the text
    # being compressed (see ``harvest_identifiers``); [] disables the guard.
    identifier_allowlist: List[str] | None = None
    # Literal strings held out of the compressor wherever they appear. The
    # MAS's routing keywords belong here: force_tokens defend them well in
    # practice, but a dropped keyword misroutes the graph and loses the run,
    # so they are worth the stronger guarantee.
    protected_literals: List[str] = field(default_factory=list)
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
        # Harvested from the whole entries, not from the payload: a symbol
        # written inside a protected code span still has to be pinned where
        # the surrounding prose names it unquoted.
        pinned = (
            self.identifier_allowlist
            if self.identifier_allowlist is not None
            else harvest_identifiers(
                entries, limit=harvested_identifier_budget(self.force_tokens)
            )
            if self.protect_identifiers
            else []
        )
        compressed = (
            self._compress_batch(payload, pinned)
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
        pattern = self._protected()
        if pattern is None:
            return [[True, text]]

        pieces: List[List] = []
        at = 0
        for span in pattern.finditer(text):
            if span.start() < at:
                # An inner match of a span already claimed by an outer one
                # (a path inside a heading, a line ref inside a code block).
                continue
            if span.start() > at:
                pieces.append([True, text[at : span.start()]])
            pieces.append([False, span.group()])
            at = span.end()
        if at < len(text):
            pieces.append([True, text[at:]])
        return pieces

    def _protected(self) -> 're.Pattern | None':
        """One alternation over every span kind this compressor holds out.

        Ordered widest first so a fenced block claims the paths and line
        references inside it rather than being cut apart by them.
        """
        key = (
            self.protect_code,
            self.protect_identifiers,
            self.protect_headings,
            tuple(self.protected_literals),
        )
        cached = _PROTECTED_CACHE.get(key)
        if cached is not None or key in _PROTECTED_CACHE:
            return cached

        parts = []
        if self.protect_code:
            parts.append(PROTECTED_SPANS.pattern)
        if self.protect_headings:
            parts.append(HEADING_SPANS.pattern)
        if self.protect_identifiers:
            parts += [PATH_SPANS.pattern, LINE_REF_SPANS.pattern]
        parts += [
            rf'\b{re.escape(literal)}\b'
            for literal in sorted(self.protected_literals, key=len, reverse=True)
        ]
        pattern = (
            re.compile(
                '|'.join(f'(?:{part})' for part in parts),
                re.MULTILINE | re.IGNORECASE,
            )
            if parts
            else None
        )
        _PROTECTED_CACHE[key] = pattern
        return pattern

    def _compress_batch(
        self, contexts: List[str], pinned: List[str] | None = None
    ) -> List[str]:
        # No question here on purpose: compress_prompt drops it before
        # reaching compress_prompt_llmlingua2 (see the class docstring).
        forced = list(self.force_tokens)
        forced += [token for token in (pinned or []) if token not in forced]
        result = self._compressor().compress_prompt(
            contexts,
            rate=self.rate,
            target_token=self.target_token,
            # Would drop whole entries and desynchronise the result list.
            use_context_level_filter=False,
            force_tokens=forced,
            force_reserve_digit=self.force_reserve_digit,
            drop_consecutive=self.drop_consecutive,
            **self.params,
        )
        texts = result.get('compressed_prompt_list')
        if texts is not None and pinned:
            texts = [unweld(text, pinned) for text in texts]
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


@dataclass
class SectionPolicy:
    """How hard to compress one labelled section of a message.

    ``label`` is written as it appears at the start of a line, with or without
    its colon: ``'THOUGHT'``, ``'EDITS:'``, ``'## Fix Plan'``. Matching
    ignores case and trailing spaces, so the ``'EDITS:           '`` an agent
    actually emits still matches ``'EDITS'``.

    ``rate`` is that section's keep-fraction, or None to pass it through
    untouched -- for a section short and decisive enough that compressing it
    buys nothing and risks everything.
    """

    label: str
    rate: float | None = None
    #: Remove the section entirely -- label and body both -- instead of
    #: compressing it. For a section that is pure ritual (a THOUGHT that only
    #: restates the plan it was handed) even a hard compression rate leaves a
    #: label with nothing under it; this drops the whole block so the next
    #: agent never sees it. Takes priority over ``rate`` when both are set.
    drop: bool = False
    #: Also lift this section into the pinned resumption state block (Z1),
    #: under this title. By default the section stays where it is as well:
    #: the block is a restatement at the top of the prompt, not a move.
    pin: str | None = None
    #: ...unless this is set, in which case the body is replaced *in place*
    #: by this line and the content lives once, in Z1. The label stays, so
    #: the scaffold is intact and the agent still finds the field where its
    #: prompt tells it to look -- only the duplicate text goes. Measured on a
    #: real restart, pinning `## Assumptions & Open Questions` verbatim *and*
    #: keeping it cost 1,264 duplicated characters, about 9% of the prompt.
    #: ``{title}`` in the text is replaced with the pin title.
    pin_pointer: str | None = None

    @property
    def key(self) -> str:
        return normalise_label(self.label)


# Marks a section's [label, body, rate] slot for wholesale removal, distinct
# from both a real rate and from None ("pass through untouched").
_DROPPED = object()


def normalise_label(label: str) -> str:
    return label.strip().rstrip(':').strip().lower()


def build_label_pattern(labels: List[str]) -> 're.Pattern | None':
    """A line-start matcher for ``labels``, longest first.

    Longest first so '## CONCERNS Review' wins over 'CONCERNS'. A markdown
    heading runs to the end of its line; a field label must be followed by its
    colon, or a line of ordinary prose that happens to start with the word is
    mistaken for a section.
    """
    if not labels:
        return None
    parts = []
    for label in sorted(labels, key=len, reverse=True):
        written = re.escape(label.strip().rstrip(':').strip())
        if label.lstrip().startswith('#'):
            parts.append(rf'[ \t]*{written}[ \t]*(?=\n|$)')
        else:
            parts.append(rf'[ \t]*{written}[ \t]*:[ \t]*')
    return re.compile('^(?:' + '|'.join(parts) + ')', re.MULTILINE | re.IGNORECASE)


@dataclass
class StructuredCompressor(ContextCompressor):
    """Compress a labelled message section by section, keeping its scaffold.

    Agent messages in a ReAct loop are not prose, they are a form: a fixed set
    of labels whose bodies mean different things and matter to different
    degrees. A Coder's ``THOUGHT`` restates the plan it was given and is nearly
    free to lose; its ``CONCERNS`` are the things the reviewing Planner has to
    adjudicate and must survive whole. Compressing the message as one blob
    applies one rate to both, and can dissolve the labels themselves -- at
    which point the next agent cannot find the fields its prompt tells it to
    read.

    So this splits an entry on its labels, compresses each body at that
    section's own rate, leaves the labels exactly as written, and puts the
    message back together in the same order. Unlabelled text -- anything
    before the first label, and any sub-heading not named in ``sections`` --
    belongs to the section it sits in and is compressed with it;
    ``default_rate`` covers a preamble, and None (the default) keeps text the
    structure does not account for verbatim.

    Which labels exist and what each is worth is a property of the target
    MAS's prompts, not of fto, so both come from configuration.
    """

    sections: List[SectionPolicy] = field(default_factory=list)
    default_rate: float | None = None
    #: Supplies the model, force_tokens, code protection and fragment floor.
    #: One clone per distinct rate; they share the loaded model because rate is
    #: not part of the cache key.
    template: LLMLinguaCompressor = field(default_factory=LLMLinguaCompressor)

    def __post_init__(self) -> None:
        self._policies = {
            s.key: (_DROPPED if s.drop else s.rate) for s in self.sections
        }
        self._pattern = self._build_pattern()

    @property
    def uses_question(self) -> bool:
        return self.template.uses_question

    def _build_pattern(self) -> 're.Pattern | None':
        return build_label_pattern([s.label for s in self.sections])

    def compress(self, contexts: List[str], question: str = '') -> CompressionResult:
        entries = [text if isinstance(text, str) else '' for text in contexts]
        plans = [self._sections(text) for text in entries]

        # A dropped section loses its label too -- it is not compressed, it
        # is gone -- so it is cleared before batching, not sent through
        # LLMLingua at some rate.
        dropped = False
        for plan in plans:
            for piece in plan:
                if piece[2] is _DROPPED:
                    piece[0] = ''
                    piece[1] = ''
                    dropped = True

        # One batched call per distinct rate rather than one per section.
        groups: Dict[float, List[tuple]] = {}
        for i, plan in enumerate(plans):
            for j, (_, body, rate) in enumerate(plan):
                if rate is not None and rate is not _DROPPED and body.strip():
                    groups.setdefault(rate, []).append((i, j))
        if not groups and not dropped:
            return CompressionResult(entries)

        # Harvested once over the whole messages, then pinned on every rate
        # clone: a symbol named in EDITS has to survive where THOUGHT mentions
        # it too, and each clone only ever sees its own group's bodies.
        pinned = (
            harvest_identifiers(
                entries,
                limit=harvested_identifier_budget(
                    getattr(self.template, 'force_tokens', [])
                ),
            )
            if getattr(self.template, 'protect_identifiers', False)
            and getattr(self.template, 'identifier_allowlist', None) is None
            else getattr(self.template, 'identifier_allowlist', None)
        )
        for rate, coords in groups.items():
            bodies = [plans[i][j][1] for i, j in coords]
            clone = replace(self.template, rate=rate)
            if pinned is not None and hasattr(clone, 'identifier_allowlist'):
                clone.identifier_allowlist = pinned
            result = clone.compress(bodies)
            for (i, j), text in zip(coords, result.texts):
                # '' from the inner compressor means "nothing was changed".
                if text:
                    plans[i][j][1] = _respace(plans[i][j][1], text)

        texts = [
            ''.join(label + body for label, body, _ in plan) if plan else ''
            for plan in plans
        ]
        return CompressionResult(
            texts,
            origin_tokens=self.template._count(entries),
            compressed_tokens=self.template._count(texts),
        )

    def _sections(self, text: str) -> List[List]:
        """[label, body, rate] triples covering the whole entry, in order."""
        if not text:
            return []
        if self._pattern is None:
            return [['', text, self.default_rate]]

        matches = list(self._pattern.finditer(text))
        if not matches:
            return [['', text, self.default_rate]]

        pieces: List[List] = []
        if matches[0].start() > 0:
            pieces.append(['', text[: matches[0].start()], self.default_rate])
        for k, match in enumerate(matches):
            end = matches[k + 1].start() if k + 1 < len(matches) else len(text)
            pieces.append(
                [
                    match.group(0),
                    text[match.end() : end],
                    self._policies.get(
                        normalise_label(match.group(0)), self.default_rate
                    ),
                ]
            )
        return pieces

    def split(self, text: str) -> List[List]:
        """``[label, body, rate]`` for every section of ``text``, in order.

        Public because the resumption state block reads the same sections
        this compresses, and the two must agree on where a section starts.
        """
        return self._sections(text)

    def labels_of(self, text: str) -> List[str]:
        """The labels this compressor recognises in ``text``, in order.

        Exposed so a caller can check a message is the shape it expected
        before handing it over.
        """
        return [label for label, _, _ in self._sections(text) if label]


def _respace(original: str, compressed: str) -> str:
    """Give a compressed body back the leading and trailing whitespace it had.

    The compressor strips both, which would otherwise weld a section onto its
    label or onto the next label's line.
    """
    lead = original[: len(original) - len(original.lstrip())]
    trail = original[len(original.rstrip()) :]
    return lead + compressed.strip() + trail
