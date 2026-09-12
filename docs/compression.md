# Context Compression — How Restart With Refined Context Works

> Covers `src/fto/recovery/compression.py` and the compression half of
> `src/fto/recovery/restart.py`. Written to be readable without knowing
> LLMLingua; the numbers in it are measured, not estimated, and come from real
> Planner/Coder runs on SWE-bench-Pro instances.

---

## 1. What this is for

When FTO restarts a faulty agent node, it has to decide what context to hand
back. There are three answers, and this document is about the third:

| Mode | What the restarted node sees |
| --- | --- |
| `RestartAllContext` | everything it saw before, unchanged |
| `RestartNoContext` | nothing |
| `RestartRefinedContext` | a **compressed** version of its history, plus its newest message intact |

The point of the middle option is to test a claim: that a restarted agent does
better with a shorter, cheaper context than with the full transcript — as long
as the short version still says the same thing. Compression is how we make the
context shorter. Most of this document is about the "still says the same
thing" part, because that is where it goes wrong.

---

## 2. The one rule

> **Compress the middle of the history. Protect both ends.**

Two kinds of message are held out:

- **The newest message** is what the node has to act on *right now* — the plan
  it must implement, or the review it must answer. `keep_last` (default 1,
  never less than 1).
- **The oldest message** is instruction-like — the task statement for a
  planner, the original plan for a coder. `keep_first` (default 1, 0 to
  disable).

A worked example. The Planner is restarted on a review turn:

```
[0]  the task specification        →  PROTECTED  (it is the instruction)
[1]  its own earlier plan          →  compressed
[2]  the Coder's report            →  compressed
[3]  the Coder's newest report     →  VERBATIM   ← what it has to answer
```

### Why the front is protected too

Because instructions are the content token pruning damages worst, and the
original design compressed them hardest. Measured on a real run: of a
17-message context, the task specification was compressed **more aggressively
than anything else** (to 62% of its length, 108 edit operations), because it is
prose while the plan is half code and code is protected. Protection was
*redirecting* the damage onto the instruction.

What that cost, verbatim from the run:

| written in the task spec | came back as |
| --- | --- |
| `DON'T have to modify the testing logic` | `'T modify testing logic` |
| `not rely on implicit zero values` | `not rely values` |
| `**Current Behavior:**` and `**Expected Behavior:**` | both → `Behavior` |
| `</uploaded_files>` | deleted |

The third row is the worst: the bug state and the target state become
indistinguishable. With `keep_first=1` all four survive intact.

> This mirrors LLMLingua's own guidance — its `compress_prompt` has separate
> `context`, `instruction` and `question` parameters precisely because
> instructions and questions are compression-sensitive while documents are
> not. FTO cannot know which message is "the instruction" in a general MAS, so
> it protects the oldest, which is the closest MAS-agnostic proxy.

**Adding structural characters to `force_tokens` does not fix this** — I tried
it. Pinning `*`, `<`, `>` and apostrophe contractions recovered **zero** of
the six lost phrases and introduced the same space-insertion artefact that
makes `.` unusable (`< tag`, `tag >`). `force_tokens` preserves *tokens*;
the damage here is the surrounding *words* being dropped, which no token
pinning can express. Not compressing the instruction is the fix.

---

## 3. How the pieces fit together

Four layers, each with one job:

```
Manager                snapshots the node's input BEFORE the fault is injected
  │                    (manager.py: snapshot() → restart.set_context())
  ▼
RestartRefinedContext  splits history from the newest message,
  │                    decides policy, remembers the result
  ▼
NodeAdapter            translates MAS messages ⇄ plain text
  │                    (context_as_list / context_from_list)
  ▼
ContextCompressor      shrinks plain text
                       (LLMLinguaCompressor, or your own)
```

Two properties of this split matter:

**The compressor never sees a message object.** It takes a list of strings and
returns a list of strings. That is why nothing in it knows or cares whether
the MAS is ChatDev or LangGraph.

**The compressor never sees the newest message.** Splitting history from the
active message is the restart layer's job, done before `compress()` is called.
If you ever call the compressor yourself, that exclusion is on you.

### What gets snapshotted

The manager takes the snapshot *before* the fault is applied, so the refined
context is built from the pristine input — not from whatever the fault left
behind. In a real run we confirmed this: a fault had rewritten the Coder's
report from 1572 to 1713 characters, and the restarted Planner received the
original 1572-character version, byte for byte.

---

## 4. What LLMLingua-2 actually does

This is the mental model that prevents most surprises.

**It is not a summarizer.** It does not rewrite, rephrase, or explain. It is a
small BERT-sized classifier that looks at each token and predicts "keep" or
"drop", then deletes the drops. The words that come out are a subset of the
words that went in, in the same order.

**And it was trained on meeting transcripts.** Both released checkpoints
(`llmlingua-2-xlm-roberta-large-meetingbank`, `llmlingua-2-bert-base-`
`multilingual-cased-meetingbank`) are distilled on MeetingBank. Go source,
XML-delimited agent prompts and API contracts are well out of that domain, so
treat compression quality on code-bearing text as **unvalidated** — which is
why the safety nets and the needle gate carry the weight here rather than
trust in the scorer. Upstream ships training scripts if you ever want a
checkpoint distilled on your own agent transcripts.

So this:

> The relevant implementation is in `contrib/snmp2cpe/pkg/cpe/cpe.go`, and the
> `Convert` function switches on `detectVendor(result)` at line 17.

becomes something like:

> relevant implementation `contrib/snmp2cpe/pkg/cpe/cpe.go` `Convert` switches
> `detectVendor(result)` line 17

Grammar is the first thing to go, because grammar is the most predictable part
of a sentence. That is usually fine — an LLM reads the second version
correctly.

**The failure mode is that it drops small words.** And some small words carry
the entire meaning of a sentence. That is the subject of the next section.

### `rate` is the fraction you keep

`rate=0.55` means "keep about 55% of the tokens". Lower is more aggressive.
(The upstream docs also mention a `ratio` parameter where larger means fewer
tokens; the installed API uses `rate`, and it is the keep-fraction.)

---

## 5. The four safety nets

Without these, compression produces text that is not merely shorter — it is
*wrong in ways that look right*. All four are on by default. The first three
try to prevent damage; the fourth checks afterwards whether they succeeded.

### 5.1 `PROTECTED_SPANS` — code never reaches the model

Anything inside backticks, or inside a fenced code block (triple backticks), is
cut out of the message before compression, held aside, and put back exactly
where it was. The model only ever sees the prose between the code.

Why this is not optional — measured on a real Planner plan, compressed without
protection:

| written | came back as |
| --- | --- |
| `` `cpe.go` `` | `` `cpe.` `` |
| `cpe:2.3:h:fortinet:%s` | `:2.:fortinet:::::::::` |
| `6.4.6` | `6. 4. 6` |
| `strings.Fields(...)` | `strings.` |
| `util.Unique(cpes)` | *(gone)* |

Every one of those is worse than deleting the sentence outright. A restarted
Planner told to read `contrib/snmp2cpe/pkg/cpe/cpe.` will call the file tool
with a path that does not exist. `:2.:fortinet:` is not a CPE string; it is
noise shaped like one.

With protection on, the same plan keeps **31 of 31** code literals.

**The limit to know:** the guarantee is by *backtick*, not by looking like
code. A literal written as bare prose is scored like any other words. In one
test `cpe:2.9:x:acme:widgetry` survived because it contains digits and
`force_reserve_digit` is on; in a real plan `cpe:2.3:o:fortinet` came back as
`:2.3` at the same rate. So if your prompts need a literal preserved, quote it
or add it to `force_tokens`. Don't rely on the digit heuristic.

### 5.2 `force_tokens` — words that must not be dropped

A list of strings the compressor is told never to remove. Two groups ship by
default, plus whatever your MAS adds.

**Negations.** Dropping "not" halves a sentence's length and inverts its
meaning. Real examples from an uncompressed-negation run:

| original | compressed | effect |
| --- | --- | --- |
| "`FortiSwitch` **must never** produce `fortios`; **only** fortigate and fortiwifi should map OS to `fortios`" | "`FortiSwitch` produce `fortios` `fortigate` `fortiwifi` map OS to `fortios`" | **says the opposite** |
| "**Do not** require `build`" | "require `build`" | **inverted** |
| "**No changes** to tests" | "changes tests" | **inverted** |
| "the minimal fix **should not** cover every product line" | "minimal fix cover every product line" | **inverted** |

`NEGATION_FORCE_TOKENS` pins `not`, `no`, `none`, `never`, `only`, `must`,
`cannot`, `without`, `except`, `unless` — and their capitalised forms, because
matching is case-sensitive and these words start sentences. With them pinned,
the same plan keeps **17 of 17** negations.

`none` deserves its own mention: `CONCERNS: none` and `CONCERNS:` mean
opposite things to a reviewing Planner.

**Structure.** `\n`, `:`, `?`, `!` keep the shape of a report readable.

**`.` is deliberately NOT forced.** Forcing it makes LLMLingua-2 treat it as
its own word and re-join it with spaces, which turns `6.4.6` into `6. 4. 6`
and `cpe.go` into `cpe. go`. This was a real bug; don't add it back.

**MAS-specific markers** belong to the experiment, not to FTO — the routing
keywords and report labels a particular workflow depends on. For Planner/Coder
that is `TASK_COMPLETE`, `READY_FOR_REVIEW`, `EDITS`, `PLAN_DEVIATIONS`,
`CONCERNS`, `REMAINING`, and the section headings. They live in
`FTOexperiments/src/config/fto_config.py`, and they **extend**
`DEFAULT_FORCE_TOKENS` rather than replacing it — replacing it silently drops
the negation protection, which is exactly what happened in one run.

<details>
<summary>A mechanism worth knowing about (checked, not currently a problem)</summary>

LLMLingua-2 handles a multi-token `force_tokens` entry by replacing it
throughout the text with an internal placeholder token, then mapping it back
afterwards. The replacement is a plain substring replace, so in principle
forcing `ACTION` also rewrites the middle of `TRANSACTION`.

Tested: `TRANSACTION`, `REACTION` and `INTERACTION` all survive intact with
`ACTION` forced, and across 237 KB of real Planner/Coder traffic there are
zero genuine collisions. No action needed — but if you add a short force token
that happens to tokenize into pieces, this is the failure mode to look for.
</details>

### 5.3 `min_fragment_chars` — don't compress scraps

Protecting code has a side effect: it chops a message into alternating prose
and code pieces. A plan dense with backticks fragments a lot. In one real
context, 6 messages became **102 prose fragments, 83 of them under 80
characters** — things like `" and "`, `" at lines "`, `" → "`.

Compressing a four-word scrap saves nothing and costs a full model pass.
`min_fragment_chars=80` keeps anything shorter verbatim. It cut compression
time on that context from **12.5s to 2.7s** with no measurable quality loss.

This threshold is also what keeps a short standalone message intact — a
history entry that is just `TASK_COMPLETE` (13 characters) is passed through
untouched.

> **Careful:** the threshold protects short *fragments*, not markers wherever
> they appear. `READY_FOR_REVIEW` at the end of a 1572-character report is
> inside a long fragment and *is* compressed — what saves it there is
> `force_tokens`, not the threshold. Both mechanisms are needed; neither
> substitutes for the other.

---

### 5.4 `CompressionValidator` — check, don't hope

The first three nets are preventive, and none of them can notice when they
were insufficient. So every compressed message is compared against its
original before being accepted, and a message that fails is **rejected — the
original is kept instead**. Fail-closed, per message, so one bad entry does
not throw away the compression of the others.

Three checks, all objective string comparisons:

| check | catches |
| --- | --- |
| **negations** | any of `not`/`no`/`none`/`never`/`neither`/`nor`/`without`/`except`/`unless`/`cannot`/`nil`/`null` and the apostrophe contractions (`don't`, `won't`, `isn't`, …) occurring fewer times than in the original |
| **code spans** | a backticked span or fenced block that did not survive verbatim — a regression check on `PROTECTED_SPANS` |
| **delimiters** | a `<tag>` present in the original and missing from the output, or a fenced block left unbalanced |

The negation check is what covers the gap `force_tokens` cannot: `DON'T` is an
apostrophe contraction, not a token that can be pinned, so pruning it to `'T`
is invisible to prevention and obvious to comparison.

Rejections are logged and recorded, so the rejection rate is measurable rather
than invisible:

```
[WARN] Refined context: rejected a compressed message and kept the original
       (negation 'not' lost (12x -> 11x); negation 'null' lost (4x -> 3x)).
```

Relax individual checks for an ablation:

```python
CompressionValidator(negations=True, code_spans=True, delimiters=False)
```

**What this revealed.** Replaying a real navidrome restart context through the
validator, the Planner's 14,914-character plan was **rejected** — it lost one
of twelve `not`s, one of four `null`s and two of twenty-five `nil`s. That
message had previously been assumed fine because its ratio looked mild (94%).
Faithful token-pruning of a code-heavy agent plan is harder than the ratio
suggests; see §8 for what that leaves.


## 6. Keeping messages aligned

The compressor returns **one compressed entry per input entry**, in the same
order. That is a hard requirement, not a convenience.

It matters because each compressed string is written back into the message it
came from. Only the text changes — the role, the `metadata['source']`, the
`keep` flag and any attachments stay exactly as they were. A compressed
Planner message is still an assistant message from the Planner. That is what
keeps the rebuilt list valid to send to a provider.

Two consequences in the code:

- `use_context_level_filter=False` is passed to LLMLingua-2 always. With it
  on, the library drops whole entries, and the returned list no longer lines
  up with the input. Alignment is worth more than that feature.
- An empty compressed entry (`''`) means "leave that message alone". This is
  how the verbatim tail is expressed, and it is also the safe outcome if a
  history entry compresses down to nothing.

Messages with nothing to rewrite never reach the model at all: empty content,
attachment-only content, and tool-call/tool-result messages, whose pairing
with each other would break if the text changed.

---

## 7. What this deliberately does not do

Being clear about the gaps is more useful than pretending they are covered.

**No question-conditioning.** LLMLingua-2 is task-agnostic. It scores the
history uniformly and has no idea what the restarted node is about to do —
compressing a plan produces the same output whether the pending review is
about a null check or a CPE string. We verified this in the library rather
than assuming it: `compress_prompt` forwards 17 parameters to the LLMLingua-2
path and silently drops 23 of them, `question` and `instruction` among them.

A compressor advertises whether it reads the question through `uses_question`,
and the restart layer only builds one when that is True. So nothing in the
code implies a conditioning that is not happening, and the log line records
which it was. **Don't add `question=` back in LLMLingua-2 mode** — it will be
discarded, and the next reader will believe it wasn't.

(LongLLMLingua — `use_llmlingua2=False` — *is* query-aware, and is the only
mode where `uses_question` is True. It needs a 7B model and ignores
`force_tokens`, which is why it is not the default.)

**No relevance-awareness at all, therefore.** The three safety nets are the
*only* thing standing between a too-aggressive rate and a mangled history.
Nothing in the pipeline notices when a rate is wrong for the content. Which is
why the rate has to be needle-tested — see §10.

**No age-tiering.** One `rate` is applied uniformly to every entry. If you
ever want older history compressed harder than recent history, build it above
this class with two compressors rather than one call:

```python
older  = LLMLinguaCompressor(rate=0.35).compress(history[:-2])
recent = LLMLinguaCompressor(rate=0.6).compress(history[-2:])
texts  = older.texts + recent.texts          # chronological order preserved
```

`rate` is not part of the model cache key, so both share the loaded model at
no extra cost in memory or load time. Needle testing so far says a flat rate
is fine for ≤5-round Planner/Coder runs, so this is not implemented.

---

## 8. What it costs

The scoring model is ~2 GB. It loads lazily on first use and is then cached
per `(model, mode, device)` for the life of the process.

Measured on a real 6-entry / ~11,900-character context:

| | GPU | CPU |
| --- | --- | --- |
| model load (cold) | 7.0s | 1.1s |
| compress, no protection | 1.0s | 8.9s |
| compress, protected, `min_fragment_chars=0` | 12.5s | 109.1s |
| compress, protected, `min_fragment_chars=80` | 2.7s | ~20s |

**The trap:** in a real GKE run with `override_gpus: null` — no GPU — one
compression took **606 seconds, 78% of the whole agent execution**. Most of
that was CPU inference plus, most likely, downloading the 2 GB model inside a
pod with a cold Hugging Face cache.

If you run in containers: bake the model into the image, and give the pod a
GPU if you can. Compression should be seconds, not minutes.

### Is it even worth it? Read this before designing an experiment

Protection lowers the compression ratio, because code no longer counts toward
the savings. Stack all four nets and the saving on a real workload gets small:

| configuration | achieved ratio | task spec |
| --- | --- | --- |
| no head protection, no validator | 86.8% | compressed to 62%, 5 of 6 key phrases lost |
| `keep_first=1` + validator (current default) | **92.5%** | intact, all 6 phrases kept |

**7.5% of tokens saved.** That is the honest number for a 17-message
Planner context on a code-heavy Go instance: the instruction is protected, the
plan is half backticked code and gets rejected by the validator for a lost
negation, and what remains to compress is short exploration notes.

Two consequences worth being explicit about:

- On this kind of workload, a refined-context restart is **barely
  distinguishable from an all-context restart**. If you are comparing the two
  arms, check `restart.compressed` and the achieved ratio per turn before
  attributing any difference to compression.
- The semantic risk is unpaid for at that ratio. The two benefit gates exist
  to say so out loud: `min_chars` skips compression when the compressible
  history is small, and `min_saving` discards a compression that saved less
  than a given fraction. Both default to off so existing configs keep their
  behaviour; `min_saving=0.2` would have skipped this turn entirely.

If you want a real token saving on this MAS, the lever is not a lower `rate` —
it is compressing the content that actually holds the token mass. A Coder
phase is hundreds of file reads and searches; that is low-sensitivity context
and it never reaches the refined restart at all, because only the agents'
final messages do.

Lowering `rate` still buys prose back, and the code stays intact either way:

| `rate` | ratio achieved | code kept | negations kept |
| --- | --- | --- | --- |
| 0.55 | 82.5% | 31/31 | 17/17 |
| 0.4 | 75.3% | 31/31 | 17/17 |
| 0.25 | 69.2% | 31/31 | 17/17 |
| 0.15 | 64.8% | 31/31 | 17/17 |

---

## 9. The same answer on every attempt

Compression runs **once per snapshot**. If `restart_count` is 3 and the node
is still faulty after two attempts, the third replays the same refined text
rather than paying for the model again.

The refined *text* is what's cached, and the message list is rebuilt fresh on
each attempt. That is deliberate: a restarted node is free to mutate the input
it was handed, and the next attempt must not inherit that.

A new snapshot (a different node, or the same node later) clears the cache and
compresses again.

---

## 10. Choosing a rate — the needle test

Since nothing in the pipeline can tell you a rate is too aggressive, the rate
is validated by experiment, not by judgement.

A needle test plants facts a restarted agent would be wrong without — a file
path, a line range, a negation, a version string — compresses, and checks they
came back. `tests/recovery/test_needle.py` is that gate. It loads the real
model, so it is opt-in:

```bash
FTO_NEEDLE_TEST=1 pytest tests/recovery/test_needle.py -v
```

Results on Planner/Coder plans:

| `rate` | verdict |
| --- | --- |
| 0.55 | every needle survived |
| 0.4 | every needle survived |
| 0.25 | `lines 87-104` came back as `87` — a line *range* lost its end |

So **0.4 is the floor** for this content. Re-run the gate if you change the
rate, the force tokens, the model, or the shape of the agents' prompts.

Two ways a needle test lies to you, both worth guarding against:

- **A needle that is already in the haystack.** If the text already contains
  the string elsewhere, "it survived" proves nothing. The committed test
  asserts every needle occurs exactly once.
- **A vacuous pass.** If nothing was actually compressed, every needle
  survives trivially. The committed test asserts the ratio really moved.

---

## 11. When compression fails

There is no safe default guess here. Silently falling back to uncompressed
text makes a refined restart indistinguishable from an all-context one, and a
benchmark that mixes the two without saying so cannot be read. So the choice
is explicit:

```python
RestartRefinedContext(on_error=CompressionFailure.RAISE)        # default
RestartRefinedContext(on_error=CompressionFailure.PASSTHROUGH)
```

- **`RAISE`** — the error propagates and the run fails. Nothing is recorded as
  refined that was not refined.
- **`PASSTHROUGH`** — falls back to the no-op `ContextCompressor`, logs a
  warning saying *"This turn is NOT refined — exclude it when comparing
  restart modes"*, and carries on.

Either way, the restart records what happened, per turn:

| field | meaning |
| --- | --- |
| `restart.compressed` | `True` compressed, `False` not, `None` not attempted yet |
| `restart.failure` | why it wasn't, when it wasn't |
| `restart.last_result` | the `CompressionResult`, including token counts |
| `restart.records` | one `MessageRecord` per message: index, policy, chars before/after, validator failures |

Read `compressed` when collecting results so a run never contains an
invisible mix. `MessageRecord.policy` is one of `verbatim-head`,
`verbatim-tail`, `compressed`, `rejected` or `unchanged`, and `.mutated`
compares chars before and after — **a `verbatim-*` record with
`mutated=True` is a bug**, since the whole point of the protected ends is that
they come back untouched. Worth asserting in any analysis script.

### A confound this is *not*

If you diff a faulty node's input against the restarted node's input, the
newest message will look shorter, and it is tempting to read that as
compression truncating a message it promised to keep verbatim. It isn't.

Verified on a real run: the faulty Planner's newest message was 5690
characters, the restarted Planner's was 5018, and the 5018 is **byte-identical
to the Coder's actual output**. The 672-character difference is the injected
FM-2.2 payload, which the snapshot legitimately excludes because the snapshot
is taken *before* the fault is applied. Nothing truncated anything.

The related worry — "a fault that gets compressed away is not a fault that was
survived" — is worth stating precisely. Every restart mode rewinds past the
fault; that is what restoring a pre-fault snapshot means, and it is equally
true of `RestartAllContext`. What FTO measures is recovery by rewind, not
survival under a persisting fault. Fault *delivery* is separately enforced:
`AegisFault.apply` raises `FaultInjectionError` if the injected text never
reached the node, so a run cannot be scored as a recovery when no fault
landed.

---

## 12. Configuration

From the experiment side (`FTOexperiments/src/config/instances/inj_refinedctx/`):

```yaml
restart:
  mode: refinedcontext
  count: 1
  keep_last: 1          # messages kept verbatim, newest first
  keep_first: 1         # messages held out at the front (the instruction)
  on_error: raise       # raise | passthrough
  min_chars: 0          # skip compression below this much history
  min_saving: 0.0       # discard a compression that saved less than this
  compression:
    use_llmlingua2: true          # false = LongLLMLingua (7B, query-aware)
    # model: microsoft/llmlingua-2-xlm-roberta-large-meetingbank
    # device: cuda                # omitted → cuda if available, else cpu
    rate: 0.55                    # fraction of tokens to keep
    protect_code: true            # hold code spans out of the compressor
    min_fragment_chars: 80        # keep prose runs shorter than this verbatim
    # force_tokens: [...]         # omitted → Planner/Coder defaults
    # target_token: -1            # hard budget; overrides rate
    # params: {}                  # anything else compress_prompt accepts
```

Two knobs exist mainly for ablations: `protect_code: false` and
`min_fragment_chars: 0` reproduce the unprotected behaviour, if you want to
measure what protection is worth.

---

## 13. Extending it

**A different compaction method.** Subclass `ContextCompressor`:

```python
class SummarizingCompressor(ContextCompressor):
    uses_question = True       # if your method reads the question

    def compress(self, contexts, question=''):
        texts = [self.llm_summarize(c, question) for c in contexts]
        return CompressionResult(texts, origin_tokens=..., compressed_tokens=...)
```

The contract is: **one entry out per entry in, same order**; `''` means "leave
that message alone". The base class is a working no-op passthrough, which also
makes it a usable stand-in for disabling compression without changing the
restart mode.

**A different MAS.** Implement two methods on your `NodeAdapter`:

- `context_as_list(context)` — the context as plain text, one entry per
  message; `''` where there is nothing a compressor may rewrite
- `context_from_list(texts, context)` — the same context rebuilt with each
  message's text replaced; `''` leaves that message untouched

The `content_text` and `replace_content_text` helpers in
`fto/adapters/node/node.py` handle the usual string and block-list content
shapes, including keeping attachments in place.

---

## 14. Gotchas, in one place

| | |
| --- | --- |
| Don't pass `question=` in LLMLingua-2 mode | It is discarded. Check `uses_question`. |
| Don't add `'.'` to `force_tokens` | Breaks `6.4.6` into `6. 4. 6`. |
| Don't *replace* `DEFAULT_FORCE_TOKENS` | Extend it, or you lose negation protection. |
| Don't assume the threshold protects markers | It protects short *fragments*. Long messages need `force_tokens`. |
| Don't assume literals are safe because they look like code | Protection is by backtick. Quote them. |
| Don't run this on CPU in a container without checking | One run spent 606s (78% of execution) compressing. |
| Don't change the rate without re-running the needle test | Nothing else will tell you it broke. |
| Don't report a run as refined without checking `restart.compressed` | `PASSTHROUGH`, skipped and rejected turns are not refined. |
| Don't try to protect prose by adding characters to `force_tokens` | Tested: recovers nothing, and `*`/`<`/`>` add spacing damage. Use `keep_first`. |
| Don't read a shorter newest message as truncation | It is the fault payload the snapshot excludes. See §11. |
| Don't assume a mild ratio means a faithful message | The plan compressed to 94% and still lost a negation. |
