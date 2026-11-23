# Implementing `RestartWithDiffSections`

## Goal

Add a fifth `Restart` strategy, `RestartWithDiffSections`, alongside
`RestartNoContext`, `RestartAllContext`, `RestartRefinedContext` and
`RestartWithDiff` (all in `src/fto/recovery/restart.py`).

It sits next to `RestartWithDiff` on purpose: same diff-file plumbing,
same "one engineered message" shape, but instead of handing back the
**entire text** of every prior message, it narrows each message down to
only its `OUTPUT` and `REPORT` labelled sections and drops everything
else — the THOUGHT/ACTION/OBSERVATION narration, any unlabelled preamble,
any other section. Nothing surviving is rewritten or compressed; a kept
section is byte-for-byte what the agent wrote. This is a **filter**, not
the token-level compression `RestartRefinedContext` does.

Where the three existing "keep some structure" modes land, for orientation:

| Mode | What survives, and how |
| --- | --- |
| `RestartRefinedContext` | *every* message, LLMLingua-compressed; only the **newest** message can be kept structure-aware (`active_compressor`, §5b of `docs/compression.md`) |
| `RestartWithDiff` | *every* message, verbatim, concatenated into one block, plus a pointer to a diff file |
| `RestartWithDiffSections` (this doc) | *every* message, but reduced to only its `OUTPUT`/`REPORT` sections, plus the same diff-file pointer |

## Read the current code first — the earlier plan for `RestartWithDiff` is stale

`docs/restart_with_diff.md` (already in this repo) describes a
`RestartWithDiff` that builds its own message inline, with `DIFF_BOUNDARY`/
`HISTORY_BOUNDARY` markers and a `no_diff_note` fallback baked into
`get_context()`. **That is not what's in `restart.py` today.** The actual
class (`src/fto/recovery/restart.py:618-638`) is:

```python
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

    def _merge_history(self, texts):
        pass
```

Message construction has been handed off entirely to an injected
`build_diff_message(idx, texts, diff_path)` callable. That callable:

- **does not exist anywhere in the codebase** — grep confirms no definition
  and no call site.
- has no default and no `build_restart` factory case (`src/local/fto_config.py`,
  `build_restart`, lines 335-378, has cases for `allcontext` / `nocontext` /
  `refinedcontext` / `None` only — none for a diff-based mode).
- is untested — no test file references `RestartWithDiff` or
  `build_diff_message`.
- leaves `no_diff_note` set in `__init__` but **unread** by `get_context()`
  — presumably meant to be used inside whatever `build_diff_message`
  eventually gets written, but currently dead.
- `_merge_history` is a stub (`pass`) called from nowhere.

`self.idx` and `self.diff_path` are populated already, unconditionally, by
`Manager._restart` (`src/fto/manager.py:234-240`):

```python
self.restart.set_idx(self.idx_step)
self.restart.set_diff(self.checkpoint.diff(self.idx_step, adapter.id))
```

So the diff-capture wiring (`Checkpoint.diff`, `set_diff`, `set_idx`) is
real and done — see `docs/restart_with_diff.md`'s "Injection index selects
the diff base" section for why `idx_step` picks `baseline_ref` vs. the
per-node snapshot ref, that part of the old doc is still accurate. **Only
the message-construction half (`build_diff_message`) is unfinished**, and
`RestartWithDiffSections` inherits that gap rather than fixing it — see
"Open questions" below.

## Step 1 — a section-extraction helper in `compression.py`

`StructuredCompressor` already has the regex machinery for splitting a
message on its labels (`compression.py:738`, `build_label_pattern`; `:862`,
`_sections`), but `_sections()` returns a `[label, body, rate]` triple for
*every* section — including unlabelled preamble and labels not in the kept
set — because it's built for "compress everything, some sections at
`rate=None`," not "keep only these labels, drop the rest." Reuse the
pattern builder, not `_sections()` itself:

```python
def keep_labelled_sections(text: str, pattern: 're.Pattern | None') -> str:
    """Only the sections `pattern` matches, label and body verbatim, in order.

    Unlike `StructuredCompressor`, nothing surviving is rewritten -- a
    kept section is exactly what was written. Everything not matched
    (unlabelled preamble, any section not in `pattern`) is dropped outright,
    not compressed down to something smaller. Used where the point is to
    isolate a decision an agent already made (its OUTPUT, its REPORT) from
    the narration around it, rather than to shrink the narration.
    """
    if not text or pattern is None:
        return ''
    matches = list(pattern.finditer(text))
    if not matches:
        return ''
    pieces = []
    for k, match in enumerate(matches):
        end = matches[k + 1].start() if k + 1 < len(matches) else len(text)
        pieces.append(text[match.start():end])
    return ''.join(pieces)
```

Put it next to `build_label_pattern` in `compression.py` and export it
alongside `StructuredCompressor` et al. in `src/fto/recovery/__init__.py`.

## Step 2 — `RestartWithDiffSections` in `restart.py`

Subclass `RestartWithDiff`, not `Restart` — the whole point is "the same
thing, plus a filter step," and subclassing means the diff plumbing
(`set_diff`, `no_diff_note`, `build_diff_message`) is inherited rather than
duplicated:

```python
class RestartWithDiffSections(RestartWithDiff):
    """`RestartWithDiff`, but every message is narrowed to a fixed set of
    labelled sections before it reaches `build_diff_message`.

    Where `RestartWithDiff` hands back the whole prior history verbatim,
    this reduces each message to only the sections that carry the decision
    already made -- by default `OUTPUT` and `REPORT`, the two labels
    `PLANNER_CODER_SECTIONS` (`src/local/fto_config.py`) documents as "the
    receiving agent's instruction" and keeps at `rate: None` for exactly
    that reason. Everything else in a message -- THOUGHT/ACTION/OBSERVATION
    narration, unlabelled preamble, any other labelled section -- is
    dropped, not compressed: the diff already carries what actually
    happened on disk, so this narrows the message history to just the
    instructions that produced it.
    """

    def __init__(
        self,
        restart_count: int = 1,
        build_diff_message: Callable | None = None,
        keep_labels: List[str] | None = None,
    ) -> None:
        super().__init__(restart_count, build_diff_message)
        self.keep_labels = keep_labels or ['OUTPUT', 'REPORT']
        self._pattern = build_label_pattern(self.keep_labels)

    def get_context(self) -> Any:
        if not self.context:
            return None
        if not self.adapter:
            raise ValueError(
                'RestartWithDiffSections needs the node adapter to read '
                'the prior context'
            )
        texts = self.adapter.context_as_list(self.context)
        filtered = [keep_labelled_sections(t, self._pattern) for t in texts]
        message = self.build_diff_message(self.idx, filtered, self.diff_path)
        return self.adapter.context_from_list([message], self.context)
```

Import `build_label_pattern` and `keep_labelled_sections` from
`fto.recovery.compression` at the top of `restart.py`, next to the existing
`CompressionFailure` / `LLMLinguaCompressor` / `guard_control_literals`
import block.

Note what did *not* change relative to `RestartWithDiff.get_context()`:
the falsy-context guard, the missing-adapter `ValueError`, and the
single-message collapse via `context_from_list([message], ...)` are all
identical. The only new line is building `filtered` before it's handed to
`build_diff_message` instead of the raw `texts`.

## Step 3 — open questions to settle before implementing

- **No protected head.** `RestartRefinedContext` exempts the oldest
  message (`keep_first`) from compression because it's instruction-like —
  the task statement or the original plan. This plan does **not** exempt
  anything: the very first message, if it has no `OUTPUT`/`REPORT` label
  at all, filters down to `''` exactly like any other unmatched message.
  That may be intentional (the diff plus later decisions already imply the
  original task), but it should be a conscious choice, not a side effect
  of reusing `RestartWithDiff`'s "no split, no head/tail" shape as-is.
  Flag this explicitly when reviewing, don't decide it silently.
- **An all-empty `texts` list is possible** — if no message in the whole
  history carries a kept label, `filtered` is a list of empty strings.
  Decide whether `build_diff_message` (whenever it's written) needs to
  special-case that the way `RestartWithDiff`'s stale plan handled "no
  diff" via `no_diff_note` — an equivalent "no matching sections" note is
  worth the same treatment, otherwise the built message reads as just the
  diff pointer with a blank history block under it.
- **`RestartWithDiff` itself is unfinished** (see above): no factory case,
  no default `build_diff_message`, no tests, dead `no_diff_note`. Since
  `RestartWithDiffSections` subclasses it and calls the identical
  `self.build_diff_message(idx, texts, diff_path)` contract — just with a
  different `texts` — whatever finishes that callable for `RestartWithDiff`
  needs to work for both, or the two modes drift into two incompatible
  message formats. Don't write a second, diff-sections-only
  `build_diff_message` implementation; finish the one contract once.
- **`keep_labels` as a flat list vs. reusing `SectionPolicy`.** An
  alternative is to accept `List[SectionPolicy]` and filter to whichever
  ones have `rate is None`, reusing `build_sections`/`PLANNER_CODER_SECTIONS`
  from `src/local/fto_config.py` directly — that keeps "what matters enough
  to survive" defined in one place instead of two (the structured
  compressor's `sections` config, and this class's own `keep_labels`). A
  plain `keep_labels: List[str] = ['OUTPUT', 'REPORT']` is simpler for a
  first version and matches exactly what was asked for; revisit only if
  the two configs are observed to drift in practice.
- **Label spelling.** `keep_labels` defaults to the exact strings
  `PLANNER_CODER_SECTIONS` uses (`'OUTPUT'`, `'REPORT'`), so
  `build_label_pattern` matches the same `LABEL:`-style fields
  `StructuredCompressor` already recognises in real Planner/Coder
  transcripts. Confirm against a real run before shipping — same
  precaution `docs/compression.md` calls out for its own section labels.

## Step 4 — wire into `build_restart` (`src/local/fto_config.py`)

Neither diff-based mode has a factory case yet. Add both together, since
`RestartWithDiffSections` needs whatever `build_diff_message` gets written
for `RestartWithDiff` regardless:

```python
        case 'withdiff':
            return RestartWithDiff(
                restart_count=count,
                build_diff_message=build_diff_message,
            )
        case 'withdiffsections':
            return RestartWithDiffSections(
                restart_count=count,
                build_diff_message=build_diff_message,
                keep_labels=restart_dict.get('keep_labels', ['OUTPUT', 'REPORT']),
            )
```

Export `RestartWithDiffSections` from `src/fto/recovery/__init__.py`
(`__all__` list and the `from .restart import ...` line) the same way
`RestartWithDiff` already is, and import it in `fto_config.py`.

## Step 5 — tests

`tests/recovery/test_restart.py`:

- `keep_labelled_sections()` unit tests: extracts `OUTPUT`/`REPORT` bodies
  verbatim, in original order, label included; drops
  THOUGHT/ACTION/OBSERVATION and any unlabelled preamble entirely; returns
  `''` for no match or empty input.
- `RestartWithDiffSections.get_context()` calls `build_diff_message` with a
  `texts` list where every entry has already been reduced to just its kept
  sections — assert on the argument the fake `build_diff_message` actually
  received, not just the final message.
- `idx` and `diff_path` reach `build_diff_message` unchanged from
  `set_idx`/`set_diff` (same assertion `RestartWithDiff`'s own tests should
  make, once those exist).
- A message with none of the kept labels present passes through as `''` in
  the `texts` list passed to `build_diff_message` — write this test to
  document the Step 3 decision explicitly, not as an incidental pass.
- Order and count of `texts` matches `context_as_list()`'s output 1:1 —
  filtering rewrites each entry in place, it does not drop or merge
  entries from the list.
- Empty `self.context` returns `None` without calling
  `context_as_list`/`build_diff_message` (mirrors `RestartWithDiff`'s
  existing early return).
- `get_context()` without a prior `set_context(..., adapter=...)` raises
  `ValueError`.

## Things to double check before calling this done

- `keep_labels` matching is case-insensitive and colon-tolerant, same as
  `StructuredCompressor`, because both go through `build_label_pattern` —
  don't hand-roll a second matcher.
- A long `OUTPUT` or `REPORT` body (e.g. one that itself contains a full
  `## Fix Plan`) survives whole under this class, since nothing here
  compresses — the combined diff-plus-filtered-history message can still
  be large. No size cap is part of this plan; add one only if a real run
  shows the resulting message is a problem, not speculatively.
- Run `uv run pytest tests/recovery/test_restart.py` (repo root
  `/home/ioanamoruz/FTO`) after implementing — confirm the existing
  `RestartRefinedContext` tests in the same file are unaffected by the new
  import in `restart.py`.
