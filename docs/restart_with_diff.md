# Implementing `RestartWithDiff`

## Goal

Add a fourth `Restart` strategy, `RestartWithDiff`, alongside the existing
`RestartNoContext`, `RestartAllContext` and `RestartRefinedContext`
(`src/fto/recovery/restart.py`).

On restart it should hand the node a **single engineered message** made of
two parts:

1. **The code diff** — a `git diff` of everything changed in the run so far.
   *Which* ref it's diffed against depends on the **injection index**
   (`fault.idx_step`, the step at which the fault was injected — see
   "Injection index selects the diff base" below): for index 1 or 2 it's the
   *clean baseline branch* (the commit the run started from, i.e.
   `GitBranchCheckpoint.baseline_ref`); for index 3 it's the *pre-fault
   per-node snapshot* (the ref `checkpoint.save(node_id)` just tagged for
   this node) instead. This must be captured **before** the pre-fault
   worktree is restored, or the faulty attempt's changes are gone before
   anyone can look at them.
2. **The prior message history, concatenated** — every message the node had
   queued before the fault, joined into one block of text (not replayed as
   separate messages the way `RestartAllContext` does).

Unlike `RestartRefinedContext`, there is no compression here: the point of
this mode is that the agent gets the *raw* diff of what actually happened on
disk plus the *raw* prior conversation, reformatted as one prompt instead of
N messages, and left to make sense of it itself.

## Where this plugs into the existing pipeline

Read `src/fto/recovery/restart.py` and `src/fto/manager.py` in full before
starting — the exact seams matter here more than for the other three modes.

```
Manager.snapshot(adapter)                              # manager.py:31
    checkpoint.save(node_id)          -> commits + tags pre-fault ref
    restart.set_context(adapter.input, adapter)         # <-- pre-fault MESSAGE snapshot

Manager.apply_fault(adapter)                            # fault mutates adapter.input
callable(...)                                           # <-- FAULTY EXECUTION, may edit files on disk

Manager._restart(callable, adapter, ...)                # manager.py:234
    for attempt in range(1, max_restarts + 1):
        checkpoint.restore(node_id)   -> `git restore --worktree` wipes
                                          the faulty attempt's file changes
        adapter.set_input(restart.get_context())         # <-- restart's OUTPUT plugs in HERE
        callable(...)                                     # re-execution
```

**The problem this creates:** `checkpoint.restore()` runs *inside* the
`_restart` loop, immediately before `get_context()`. By the time
`RestartWithDiff.get_context()` is called, the working tree has already been
rewound and the diff the agent is supposed to see no longer exists on disk.
So the diff has to be captured once, **before** the retry loop's first
`checkpoint.restore()` call — at the top of `Manager._restart`, not inside
`Restart.get_context()`.

This means `RestartWithDiff` needs a second setter, parallel to
`set_context`, that `Manager` calls with the diff text once it has it.
`Restart` gains a no-op `set_diff` on the base class so `Manager` can call it
unconditionally on every restart mode, exactly the way it already calls
`set_context` unconditionally regardless of which subclass is active.

## Injection index selects the diff base

`fault.idx_step` is the node-sequence step the fault was injected at
(`Manager.idx_step` at injection time — see `manager.py:51`,
`_fault_due()`). This run's fault configs only ever place it at 1, 2 or 3
(**it cannot be higher than 3**), and that index changes what "the diff"
means:

- **index 1 or 2** — diff against `baseline_ref`, the same as the rest of
  this plan describes: everything the run has done so far, from the clean
  start of the whole workflow.
- **index 3** — diff against the **per-node snapshot** instead, i.e. the ref
  `checkpoint.save(node_id)` tagged for *this* node right before the fault
  was applied (`Manager.snapshot()`, `manager.py:31` — the same ref
  `checkpoint.restore(node_id)` later resets to). This is a narrower diff:
  only what changed from this node's own pre-fault state, not the whole run.

Why the split: at index 1–2 there's little or no prior run history for a
per-node snapshot to usefully isolate — the baseline and the snapshot are
close to the same point, so diffing against the baseline is both correct and
simpler. By index 3 enough of the run has happened that "everything since
the very start" stops being what a restarted node needs to reorient against;
what matters is what *this* node's faulty attempt did, which the per-node
snapshot isolates and the baseline diff would bury in earlier nodes' changes.

Concretely, `GitBranchCheckpoint.diff()` needs to know both `idx_step` and
`node_id` to pick the right ref, and should reject an `idx_step` above 3
outright (a config bug, not a case to silently reinterpret) rather than
falling back to either ref. `Manager._restart` is the caller that has both
values on hand (`self.fault.idx_step` and `adapter.id`) — see Step 4's
updated call site.

## Step 1 — teach `Checkpoint` to produce a diff **file**

**Deviation from a first draft of this plan:** `diff()` does not return the
diff as a string. It writes the diff to a file on disk and returns the
`Path` (or `None`). A long-running MAS with many file-touching nodes can
produce a diff too large to want to carry around in memory or inline into a
message — writing it out lets the restarted node (or a human inspecting the
run) reach for it only if and when it needs it, instead of it always paying
for that many tokens up front. This also matches the task's framing:
the restarted node gets "the message history, as well as an available diff
file" — the diff is a resource made available, not text stitched into the
prompt.

`src/fto/recovery/checkpoint.py` already holds the git plumbing
(`repo_path`, `_git()`, `baseline_ref`). Add a `diff()` method there rather
than shelling out again elsewhere.

```python
import os
import tempfile
from pathlib import Path


class Checkpoint:
    ...
    def diff(self, idx_step: int | None, node_id: str) -> Path | None:
        return None


class GitBranchCheckpoint(Checkpoint):
    ...
    def diff(self, idx_step: int | None, node_id: str) -> Path | None:
        """Write everything changed since the appropriate ref to a file.

        Which ref is "appropriate" depends on the injection index (see
        "Injection index selects the diff base" above):

        - index 1, index 2, or ``None`` (no injection index -- a
          detection-only restart with no fault involved) -> ``baseline_ref``,
          the commit taken before the very first node ran. A restart needs
          to see everything the run has done so far, not just what the
          faulty node itself touched.
        - index 3 -> the per-node snapshot ref this node's own
          ``save(node_id)`` tagged, i.e. only what changed since *this*
          node's own pre-fault state.

        ``idx_step`` cannot be greater than 3 for this run's fault configs;
        that's a configuration bug, so it's raised rather than silently
        clamped or defaulted to a ref.

        Returns ``None`` when the chosen ref doesn't exist yet or nothing has
        changed since it, so callers can tell "no diff" apart from "diff not
        asked for" without also having to open a file to find out it was
        empty.

        Must be called before ``restore()`` wipes the faulty attempt's
        worktree changes back to the pre-fault state.
        """
        if idx_step is not None and idx_step > 3:
            raise ValueError(
                f'idx_step {idx_step} is not supported; the injection index '
                f'cannot be higher than 3.'
            )
        ref = self._ref(node_id) if idx_step == 3 else self.baseline_ref
        if not ref or not self._ref_exists(ref):
            return None
        result = self._git('diff', ref)
        if not result.stdout.strip():
            return None
        fd, path = tempfile.mkstemp(
            prefix=f'{self.branch_prefix}-diff-', suffix='.patch'
        )
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
        return Path(path)
```

`_ref_exists` is new: `baseline_ref` was previously guarded with
`getattr(self, 'baseline_ref', None)` because it's only set once
`save_baseline()` has run, but the index-3 ref (`self._ref(node_id)`) is a
plain string built from `node_id` regardless of whether `save(node_id)` was
ever actually called, so the old guard doesn't cover it. Add a small helper
next to `_is_git_repo`/`_is_dirty` that checks the branch actually exists
(e.g. `git rev-parse --verify --quiet <ref>`, treating a non-zero exit as
"doesn't exist" rather than raising through `_git()`), and use it for both
refs so an unset baseline and a never-`save`d node snapshot fail the same
safe way.

Notes:
- `git diff <ref>` (no `--cached`) compares `<ref>` against the **working
  tree**, so it picks up both prior nodes' committed changes and the current
  faulty node's uncommitted edits in one call — exactly "all changes made"
  (relative to whichever ref index 1–2 vs. 3 selects).
- `baseline_ref` is only set once `save_baseline()` has run (start of the
  whole workflow run); the per-node ref only exists once `save(node_id)` has
  run for that node. Both are covered by `_ref_exists()` — guard against
  either being unset/never-tagged (e.g. a unit test that never calls
  `save_baseline`/`save`) by returning `None`, not by raising.
- `idx_step > 3` is the one case that *does* raise (`ValueError`) rather than
  returning `None` — it means the caller is misconfigured, not that there's
  nothing to diff.
- `_git()` already raises `RuntimeError` with stderr attached on a non-zero
  exit, which is what you want for the actual `diff` call — don't swallow it.
  `_ref_exists()` is the one place that deliberately turns a `git`
  non-zero exit into `False` instead of letting it propagate.
- Writing to a system temp file (rather than somewhere under `repo_path`)
  keeps the diff file out of `git status`/`_is_dirty()` entirely — no need to
  extend `_FTO_EXCLUDE_PATTERNS` for it, and no risk of the diff file itself
  showing up in a later diff.
- Every downstream reference to "the diff" in this plan (the message text
  `RestartWithDiff` builds, `Manager._restart`'s call to `set_diff`, the
  tests in Step 7) should be read as "the diff **path**", not the diff text.
  `Restart.set_diff` / `RestartWithDiff.set_diff` take the `Path | None`
  `Checkpoint.diff()` returns, store it as-is, and `RestartWithDiff` puts a
  line in its engineered message pointing at that path (e.g. `A diff of
  everything changed in the repository since the run started is available
  at: <path>`) instead of embedding the diff's contents. `no_diff_note` is
  still used verbatim when `diff_path` is `None`.

## Step 2 — add `set_diff` to the `Restart` base class

`src/fto/recovery/restart.py`:

```python
class Restart:
    def __init__(self, restart_count: int = 1) -> None:
        self.restart_count = restart_count
        self.context: Any = None
        self.adapter: 'NodeAdapter | None' = None

    def set_context(self, context: Any, adapter: 'NodeAdapter | None' = None) -> None:
        self.context = deepcopy(context)
        self.adapter = adapter

    def set_diff(self, diff_path: 'Path | None') -> None:
        """Path to the diff file of everything changed so far, captured before
        the worktree is restored (see Step 1's deviation note: this is a
        `Path`, not diff text).

        A no-op on the base class: only a mode that actually uses the diff
        needs to override it. ``Manager`` calls this unconditionally on every
        restart, the same way it calls ``set_context`` unconditionally.
        """
        pass

    def get_context(self) -> Any:
        pass
```

## Step 3 — `RestartWithDiff`

Same file, after `RestartAllContext`/`RestartNoContext`:

```python
DIFF_BOUNDARY = '── DIFF OF ALL CHANGES SINCE THE RUN STARTED ──'
HISTORY_BOUNDARY = '── PRIOR MESSAGES BEFORE THE FAULTY ATTEMPT ──'


class RestartWithDiff(Restart):
    """Restart on one engineered message: prior history plus a pointer to the
    accumulated code diff, written out to a file rather than inlined.

    Unlike ``RestartRefinedContext`` nothing here is compressed or split into
    a protected head/tail — the raw history is handed back verbatim, joined
    into a single message instead of replayed as separate ones. The diff
    itself is not embedded in the message text: it is written to disk by
    ``Checkpoint.diff()`` (see Step 1) and the restarted node is told where to
    find it. The restarted node has to reconcile "what I was asked to do"
    against "what the repo actually looks like right now", which is exactly
    the information a faulty attempt's own message history doesn't carry.
    """

    def __init__(self, restart_count: int = 1, no_diff_note: str | None = None) -> None:
        super().__init__(restart_count)
        self.diff_path: 'Path | None' = None
        # What to say instead of pointing at a diff file when there is
        # nothing to show (first-ever node in the run, no changes yet, or a
        # non-git working tree). Silence here reads as "nothing changed",
        # which is a claim, not an absence.
        self.no_diff_note = no_diff_note or 'No changes were made to the repository yet.'

    def set_diff(self, diff_path: 'Path | None') -> None:
        self.diff_path = diff_path or None

    def get_context(self) -> Any:
        if not self.context:
            return self.context
        if self.adapter is None:
            raise ValueError(
                'RestartWithDiff needs the node adapter to read the prior '
                'context; pass it to set_context().'
            )

        texts = self.adapter.context_as_list(self.context)
        history = '\n\n'.join(text for text in texts if text and text.strip())
        diff_block = (
            f'A diff of everything changed in the repository since the run '
            f'started is available at: {self.diff_path}'
            if self.diff_path
            else self.no_diff_note
        )

        message = (
            f'{DIFF_BOUNDARY}\n\n{diff_block}\n\n'
            f'{HISTORY_BOUNDARY}\n\n{history}'
        )
        return self.adapter.context_from_list([message], self.context)
```

Design choices worth keeping, and why:
- **A path, not the diff text** — see Step 1. `set_diff` stores whatever
  `Checkpoint.diff()` returned (a `Path`, or `None`), and the message names
  that path rather than quoting the file's contents.
- **One message, not many** — `context_from_list([message], self.context)`
  collapses everything to a single entry, matching the task's "engineered
  into a single message" requirement and the `NodeAdapter` contract (it
  rebuilds *a* context from a text list; a one-element list is valid input).
- **No compression, no protected head/tail** — this mode is deliberately the
  "just show it everything" baseline against `RestartRefinedContext`'s
  "compress and pin" approach. Don't borrow `_split`/compression logic here;
  that would just reimplement `RestartRefinedContext` under a different name.
- **Boundary markers** — reuse the pattern from `RESUMPTION_BOUNDARY` /
  `TAIL_BOUNDARY` already in this file: a plain sentinel line so a human or
  model reading the message can tell where the diff ends and the historical
  conversation begins, without relying on subtler formatting.
- **`get_context()` guards `self.context` falsy** exactly like the other two
  simple modes — an empty snapshot restarts to an empty context, diff or no
  diff, rather than fabricating a message around nothing.

## Step 4 — wire the diff capture into `Manager`

`src/fto/manager.py`, `_restart` (around line 234): capture the diff **once**,
before the retry loop's first `checkpoint.restore()` call, and push it into
the restart strategy via the new `set_diff` hook:

```python
def _restart(self, callable, adapter, *args, token=None, **kwargs):
    """Re-execute the node, up to ``restart.restart_count`` times for THIS
    node invocation."""
    max_restarts = getattr(self.restart, 'restart_count', 1)
    if self.checkpoint:
        # Must run before the loop's first restore() wipes the faulty
        # attempt's on-disk changes back to the pre-fault state. idx_step
        # picks baseline vs. per-node snapshot as the diff base -- see
        # "Injection index selects the diff base". self.fault can be None
        # (a detection-only restart with no injected fault), so it isn't
        # read unconditionally.
        idx_step = self.fault.idx_step if self.fault else None
        self.restart.set_diff(self.checkpoint.diff(idx_step, adapter.id))
    result = None
    for attempt in range(1, max_restarts + 1):
        ...  # unchanged
```

**`self.fault` is not always set here — don't assume it is.** `_restart` can
be reached two ways (`manager.py:96`, `manager.py:121`): after a fault
injection (`self.fault` is a real `Fault`), *or* purely because
`self.observer` flagged a detection on an unfaulted node
(`_run_observed`'s `restartable = fault_applied or (faults and
self.observer.should_restart(faults))`) — and `Manager.__init__` accepts
`fault: Fault | None`, so a detection-only setup runs with `self.fault is
None`. Reading `self.fault.idx_step` unconditionally would raise
`AttributeError` in that case, hence the `if self.fault else None` above.

Decide (as part of this step, not left implicit) what `Checkpoint.diff` does
with `idx_step=None` — there is no injection index to key off when recovery
was triggered by detection alone. Diffing against `baseline_ref` in that
case (treat "no index" the same as index 1–2, as Step 1's implementation
above already does) is the reasonable default: it's the same "show
everything the run has done so far" behavior `RestartWithDiff` uses
elsewhere, and nothing in the spec says a detection-triggered restart should
see a narrower diff than an injection-triggered one at index 1–2. Make this
an explicit branch in `diff()`, not a fallthrough that happens to work.

This is safe to call unconditionally for every restart mode (mirrors
`set_context`'s unconditional call in `snapshot()`), because `set_diff` is a
no-op on the base `Restart` and on the three pre-existing modes — only
`RestartWithDiff` does anything with the value. No `isinstance` check needed,
and no other mode's behaviour changes.

Also double check `Manager.__init__` still takes `checkpoint` as an already
existing constructor arg (it does, `manager.py:19`) — no signature change
needed there.

## Step 5 — wire the config

`FTOexperiments/src/config/fto_config.py`, `build_restart` (around line
301), add a new `case`:

```python
        case 'withdiff':
            return RestartWithDiff(
                restart_count=count,
                no_diff_note=restart_dict.get('no_diff_note'),
            )
```

And update the import at the top of that file to include `RestartWithDiff`
from `fto.recovery`.

Also export it from the framework's `src/fto/recovery/__init__.py`
(`__all__` list and the `from .restart import ...` line), the same way
`RestartAllContext` etc. are exported — otherwise `fto_config.py` can't
import it.

## Step 6 — add an example instance config

Mirror the existing `inj_allctx` / `inj_noctx` / `inj_refinedctx` folders in
`FTOexperiments/src/config/instances/`. Copy
`inj_refinedctx/aegis_FM22.yaml`, but strip the compression/resumption knobs
this mode doesn't use, and require a git checkpoint since the diff is
meaningless without one:

```yaml
# src/config/instances/inj_withdiff/aegis_FM22.yaml
fault:
  type: aegis
  node_id: Coder
  mode: FM-2.2
  # ... (unchanged from the other inj_* variants)
restart:
  mode: withdiff
  count: 1
  no_diff_note: "No changes were made to the repository yet."
checkpoint:
  mode: gitbranch   # required: RestartWithDiff needs baseline_ref to exist
```

If `checkpoint` is missing or not `gitbranch`, `Manager.checkpoint` is
`None`/a different type, `self.checkpoint.diff()` is never called, and
`RestartWithDiff.get_context()` falls back to `self.no_diff_note` — verify
this fallback explicitly in a test (Step 7) rather than assuming it.

## Step 7 — tests

Add to `tests/recovery/test_restart.py`, following the existing
`TestRestartAllContext` / `TestRestartNoContext` classes as the template
(they already show the `FakeAdapter`/`FakeMessage` fixtures at the top of the
file):

- `set_context` + `set_diff` + `get_context()` produces one message
  containing both boundary markers, the diff **path** (not diff content), and
  every prior message's text joined together.
- Order of prior messages in the concatenated history matches
  `context_as_list`'s order (don't accidentally reverse or dedupe).
- Empty `self.context` returns `self.context` unchanged, without calling
  `context_as_list`/`context_from_list` (mirrors the existing "no context"
  early-return tests for the other modes).
- No diff set (`set_diff` never called, or called with `None`/`''`) falls
  back to `no_diff_note` in the message body.
- `get_context()` without a prior `set_context(..., adapter=...)` raises
  `ValueError` (mirrors `RestartRefinedContext`'s equivalent check).

Add to `tests/test_manager.py`:

- A restart with a real (or fake) `Checkpoint` whose `diff()` is stubbed
  confirms `Manager._restart` calls `checkpoint.diff(idx_step, node_id)` with
  `self.fault.idx_step` and `adapter.id`, and `restart.set_diff(...)` with
  its result, exactly once per node-level restart, **before** the first
  `checkpoint.restore()` call for that node (assert call order and
  arguments, not just call count — the ordering is the entire point of this
  design).
- A restart with `checkpoint=None` never calls `set_diff` and doesn't error.
- A restart triggered with `self.fault is None` (detection-only, no
  injection) calls `checkpoint.diff(None, node_id)` rather than raising
  `AttributeError` on `self.fault.idx_step`.

Add to `tests/recovery/test_checkpoint.py`:

- `Checkpoint.diff(idx_step, node_id)` (base class) returns `None` regardless
  of the arguments given.
- `GitBranchCheckpoint.diff(idx_step, node_id)` at index 1 or 2 returns
  `None` when no baseline has been saved, and `None` again when a baseline
  exists but nothing has changed since it (don't write an empty file in
  either case).
- A real integration test at index 1 (or 2 — behavior should be identical):
  init a temp git repo, `save_baseline()`, make a file edit, assert
  `checkpoint.diff(1, node_id)` returns a `Path` whose contents contain that
  edit; then `save(node_id)` + another edit + assert `diff(1, node_id)` now
  returns a file containing *both* edits (cumulative against baseline, not
  against the last node checkpoint). Also cover an uncommitted
  (never-`save`d) worktree edit showing up in the diff.
- A real integration test at index 3: `save_baseline()`, edit + `save('n1')`
  (the pre-fault snapshot for the node under fault), edit again, and assert
  `diff(3, 'n1')` contains only the *second* edit — not the one already
  captured by `save_baseline()` — proving it diffs against the per-node
  snapshot and not the baseline.
- At index 3, `diff(3, node_id)` returns `None` when `save(node_id)` was
  never called for that `node_id` (no snapshot ref exists yet), even if
  `baseline_ref` exists and has changes against it — the two refs are not
  interchangeable fallbacks for each other.
- `diff(None, node_id)` behaves like index 1/2 (diffs against `baseline_ref`)
  — covers the detection-only restart path from `test_manager.py` above.
- `diff(4, node_id)` (and any `idx_step > 3`) raises `ValueError` without
  touching git or writing a file.

## Things to double check before calling this done

- Writing the diff to a file (Step 1) sidesteps the original worry about
  `git diff <ref>` output blowing the target model's context window — the
  message only ever contains a path. It does not sidestep the diff file
  itself being large on disk; no truncation is needed for that, since nobody
  reads it until something opens the file.
- These diff files accumulate under the system temp directory across restarts
  and are never cleaned up by `Checkpoint`/`Manager`. Decide whether that's
  acceptable (they're small relative to a typical run, and the OS temp dir
  gets swept eventually) or whether `Manager`/the caller should unlink them
  once a run finishes. Don't add cleanup speculatively — only if a real run
  shows it matters.
- Binary file changes show up in `git diff` as `Binary files a/... and b/...
  differ` — that's fine to pass through as-is into the file, no special
  handling needed.
- A node that is the very first one in the run has an empty
  `context_as_list` (or a single task-statement message) and possibly no
  diff yet (`diff()` returns `None`, nothing committed) — confirm the "first
  turn" case reads sensibly rather than producing a message that's just two
  boundary lines with nothing under either.
- Run `uv run pytest tests/recovery/test_restart.py tests/test_manager.py`
  (repo root `/home/ioanamoruz/FTO`) after implementing, not just on the new
  tests — `RestartRefinedContext`'s tests exercise the same `Manager._restart`
  call path and must keep passing unchanged.
