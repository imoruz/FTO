# FTO Observer Layer — Architecture & Implementation Guide

> Status: design proposal (v2). Adds a fault/timeout **detection** layer to FTO,
> maps detections onto **ATP** status codes, and drives the existing restart
> machinery from ATP recovery semantics.
>
> **What changed in v2.** The first draft observed only at the *node boundary*
> (inspect adapter state before/after, catch exceptions around the node call).
> After reading how ChatDev and LangGraph nodes actually execute, that is
> provably insufficient: the real LLM and tool calls happen in an **inner
> agentic loop inside the node**, and that loop **swallows** the very faults we
> want to detect. v2 reworks the observer to instrument the inner call sites —
> the same sites `instrumentation.py` already patches for OTel — so detection
> sees inner failures *before the framework hides them*.

---

## 1. Background & problem statement

Today FTO is a fault **injection** harness. `Manager.make_supervised_callable`
(`src/fto/manager.py`) wraps a single framework *node-dispatch* function and, on
the configured agent step, *injects* a fault (prompt corruption, AEGIS, OTel
infra), suppresses edges, then optionally restores a checkpoint and restarts the
node. Concretely, the wrapped dispatch function is:

| Framework  | Wrapped node-dispatch (`Supervisor.start`)      | Adapter built from        |
| ---------- | ----------------------------------------------- | ------------------------- |
| ChatDev    | `GraphExecutor._execute_node(node)`             | `ChatDevNodeAdapter(node)`|
| LangGraph  | `langgraph.pregel._runner.run_with_retry(task)` | `LangGraphNodeAdapter(task, spec)` |

What is missing is a fault **observation** layer: a component that watches real
node execution and decides, after the fact, *whether a fault actually occurred*
and what to do about it.

### 1.1 The gap the wrapper cannot see (the core problem)

`make_supervised_callable` wraps the node-dispatch function, so it sees one
node go in and one node come out. But for the planner–coder workflows the
**actual** work — every LLM call and every tool call — happens in an *inner
agentic loop nested inside that single dispatch*. And that loop is written to
**absorb faults locally** so the agent can keep going:

In LangGraph (`experiments/langgraph/instances/planner_coder.py`,
`AgentStrategy._run_agentic_loop`):

```python
while iteration < max_iterations:                 # max_iterations = 200
    try:
        response = llm_with_tools.invoke(messages) # LLM call
    except Exception as e:
        new_messages.append(AIMessage(content=str(e)))   # <-- LLM error swallowed
        break                                            #     node returns "ok"
    ...
    for tc in tool_calls:
        try:
            result_content = tool_fn.invoke(tc['args'])  # tool call
            consecutive_errors[tool_name] = 0
        except Exception as exc:
            result_content = f"ERROR calling '{tool_name}' ...: {exc} ..."  # <-- tool
            is_error = True                                                 #     error
        if is_error:                                                        #     swallowed
            consecutive_errors[tool_name] += 1
            if consecutive_errors[tool_name] >= max_consecutive_errors:     # = 5
                banned_tools.add(tool_name)        # tool silently disabled
        messages.append(ToolMessage(content=str(result_content), ...))
else:
    fto_logger.log_event('... hit max_iterations, forcing exit')  # silent loop exit
```

ChatDev is structurally identical: the dispatch (`_execute_node`) drives an
`AgentNodeExecutor` whose inner LLM call is `_invoke_provider` and whose tool
calls go through `ToolManager.execute_tool` — both of which can fail *inside*
the node while the node still returns.

**Consequences for the boundary-only observer in v1:**

1. **Exceptions never reach the boundary.** An inner LLM/tool exception is
   caught and turned into a message; the node returns normally. A
   `classify_exception` hook wrapped around the node dispatch never fires.
2. **Adapter state doesn't reveal it.** The adapter exposes node *input*
   (`task.input.messages` / `node.input`) and `last_message`. The inner error
   `ToolMessage`s, the banned tools, and the "hit max_iterations" exit are
   internal to the loop and are not visible through the adapter at the boundary.
3. **Self-recovery hides degradation.** A node that burned 200 iterations, got
   a tool banned, and produced a degraded answer looks *successful* from
   outside.

So the observer must **descend into the node** and watch the inner LLM and tool
calls themselves. That is the whole point of v2.

### 1.2 The two fault shapes (unchanged from v1)

1. **Context faults** — the call "succeeded" (no exception) but a payload is
   corrupted/unsafe: a prompt injection landed, memory was poisoned, an output
   violates its schema. Detected by *inspecting payloads*.
2. **Exception faults** — a call raised (or hung): timeout, runtime error, bad
   upstream response, transport drop, sandbox failure. Detected by *catching*
   (or by a watchdog when a call hangs).

v2's change is *where* we look for both: not only at the node boundary, but at
each inner LLM/tool call, **before** `_run_agentic_loop` converts the failure
into a message and moves on.

The observer classifies both shapes into one typed vocabulary — **ATP, the
Agentic Transfer Protocol** status codes (https://atp-protocol.com) — then,
when recovery is warranted **and a restart is configured**, triggers the
existing restart flow.

---

## 2. Why ATP is the right backbone

Each ATP status carries four fields we care about:

- `code` — a stable integer (e.g. `504`, `650`).
- `name` — human label (e.g. `Gateway or Dependency Timeout`).
- `recoveryHint` — one of a small fixed action vocabulary (below).
- `faultTags` — cross-references to MAST / AgentFail / TRAIL / OWASP taxonomies,
  which dovetail with FTO's existing AEGIS `FMErrorType` and OTel fault specs.

The **recovery-hint vocabulary** decouples "what went wrong" from "what to do":

| recoveryHint        | Meaning                          | FTO action                                   |
| ------------------- | -------------------------------- | -------------------------------------------- |
| `continue`          | In-progress, not terminal        | no-op (let it run)                           |
| `advance`           | Success, move on                 | no-op                                        |
| `reroute`           | Pick a different handler         | no-op for now (FTO can't reroute) → log only |
| `retry_backoff`     | Transient, retry                 | **trigger restart**                          |
| `repair_request`    | Caller/input is malformed        | **trigger restart** (with repaired context)  |
| `verify_or_repair`  | Semantically wrong, re-derive    | **trigger restart**                          |
| `human_review`      | Escalate to a human              | no-op for now → log + flag                   |
| `abort`             | Unsafe / unrecoverable, stop     | no restart; surface/raise                    |

The observer's core decision reduces to: **detect → ATP code → recoveryHint →
(if hint ∈ {retry_backoff, repair_request, verify_or_repair} and a `Restart` is
configured) → run the existing restart flow.** A crucial v2 refinement: a single
inner tool error that the agent *recovered from on its own* should map to
`continue` (log only). The implemented decision policy is **first fault detected
wins** (`Observer.first_fault`, §5.1): the earliest detection in the node becomes
the verdict and the Manager acts on its recovery hint, with no aggregation or
thresholding.

### 2.1 ATP codes FTO will actually use

A curated subset (the full registry is ~90 codes; FTO only needs the ones it can
plausibly *detect* at an inner call site or node boundary and *act on*).

**Exception-shaped faults (caught at an inner call site, or watchdog-tripped):**

| code | name                            | recoveryHint    | typical trigger (where observed)                         |
| ---- | ------------------------------- | --------------- | -------------------------------------------------------- |
| 408  | Caller-side Timeout             | retry_backoff   | required input/continuation not in time                  |
| 500  | Internal Agent Runtime Error    | retry_backoff   | unhandled exception in an inner LLM/tool call            |
| 502  | Bad Upstream Response           | retry_backoff   | tool/model returned corrupt data                         |
| 503  | Service Unavailable             | retry_backoff   | provider/tool down/overloaded                            |
| 504  | Gateway or Dependency Timeout   | retry_backoff   | **watchdog deadline exceeded** (per-call or per-node)    |
| 507  | Resource Exhausted              | retry_backoff   | OOM / token / quota exhaustion                           |
| 508  | Runtime Loop Detected           | abort           | **`max_iterations` hit / tool banned after N failures**  |
| 520  | Transport Fault                 | retry_backoff   | message drop/reorder/partition                           |
| 522  | Sandbox or Environment Failure  | retry_backoff   | missing pkg / interpreter failure in a tool              |
| 529  | Provider-side Rate Limit        | retry_backoff   | provider throttle (respect retry_after)                  |

**Context-shaped faults (detected by inspecting an inner payload or node state):**

| code | name                            | recoveryHint    | typical trigger                          |
| ---- | ------------------------------- | --------------- | ---------------------------------------- |
| 413  | Context Too Large               | repair_request  | prompt/memory exceeds budget             |
| 460  | Invalid Agent Output Format     | repair_request  | LLM/tool output violates schema/parser   |
| 462  | Unsafe Input Detected           | abort           | adversarial/injected input *pre*-exec    |
| 601  | Hallucinated Content            | verify_or_repair| unsupported factual/code claims          |
| 640  | Context Loss or Amnesia         | verify_or_repair| forgot relevant history/state            |
| 642  | Memory Poisoning Propagated     | abort           | compromised memory used downstream       |
| 650  | Prompt Injection Succeeded      | abort           | injected instruction changed behavior    |

> Note the asymmetry: some context faults (462, 642, 650) and the loop-detector
> (508) carry `abort`, meaning "do not silently restart — surface it." The
> observer still *reports* them; it just doesn't auto-restart. This is exactly
> the behavior you want for the prompt-injection faults FTO itself injects.

---

## 3. Design decisions

### 3.1 Extend the existing wrapper, don't add a parallel one (unchanged)

Observation and injection share *the same recovery machinery* — `Restart`,
`Checkpoint`, `EdgeSuppressor`, `idx_step` bookkeeping, logging, and the
`record_injection` → OTel bridge. So keep **one** `Manager`-owned node wrapper
with gated phases (injection, observation, recovery) rather than a second
`make_observed_callable`. The observer's *detection logic* lives in its own
module (`observer.py`); `Manager` orchestrates, `Observer` classifies.

### 3.2 Observe at two levels, with the inner level as a side channel (new)

The node wrapper alone cannot see inner calls (§1.1). But the inner calls happen
deep inside framework code, in methods we don't call directly — we can't thread
a return value back up through `_run_agentic_loop`. The proven technique, used
already by `instrumentation.py`, is to **monkeypatch the inner call sites** and
record observations into a **side channel** that the node wrapper drains
afterwards. v2 adopts exactly this:

- **Inner probes** wrap the inner LLM and tool call sites (the same methods
  `apply_patches` wraps). On each inner call they: run it, catch any exception
  (record a `Detection`, then **re-raise** so the framework's own handling is
  unchanged), and inspect the result payload (record a `Detection` if it looks
  like an error/corruption). They never alter control flow — observation is
  purely additive, preserving FTO's "injection-only behaves exactly as today"
  guarantee.
- **A `DetectionSink`** (context-aware, mirroring `injection_notice` and the
  `_session_steps` ContextVar) is where probes write. The node wrapper opens a
  per-step scope, lets the node run, then drains the scope.
- **Node-boundary inspection** (the v1 mechanism) is *kept* for context faults
  visible in adapter state (e.g. the injected prompt marker the FTO test loop
  checks) and for the rare exception that does escape the node.

```
                       ┌──────────────── node dispatch (wrapped by Manager) ───────────────┐
 Manager.wrapped ──▶   │  _execute_node / run_with_retry                                    │
   open scope ─────────┤    └─ inner agentic loop                                            │
                       │         ├─ LLM call  ◀── Probe(_invoke_provider / _generate)        │
                       │         │     catch+record(5xx) → re-raise ; inspect output (460/601)│
                       │         └─ tool call ◀── Probe(execute_tool / BaseTool.invoke)       │
                       │               catch+record(502/522) → re-raise ; inspect result      │
   drain scope ◀───────┤    (loop swallows the re-raised exc as before; Detections remain)    │
   + boundary inspect  └──────────────────────────────────────────────────────────────────────┘
   + first_fault → decide restart (existing recovery flow)
```

### 3.3 Phase model of the unified wrapper

```
wrapped(*args, **kwargs):
  adapter = make_adapter(...)

  ┌─ OBSERVER DISABLED (self.observer is None) — today's behavior, byte-for-byte ─┐
  │   if not adapter.is_agent: return callable(...)      # non-agent pass-through  │
  │   idx_step += 1                                                               │
  │   INJECTION PHASE (only when self.fault matches this step):                    │
  │     checkpoint.save / restart.set_context / fault.apply / record_inject / rerun│
  └────────────────────────────────────────────────────────────────────────────────┘

  ┌─ OBSERVER ENABLED (self.observer is set) — OBSERVE EVERY NODE ──────────────────┐
  │   is_agent = adapter.is_agent                                                    │
  │   if is_agent: idx_step += 1        # agent-step counter gates injection only    │
  │   node_seq += 1                     # per-node counter for scope attribution     │
  │                                                                                  │
  │   INJECTION (agent + self.fault matches this step only):                          │
  │     if is_agent: checkpoint.save / restart.set_context   # snapshot agents only  │
  │     if fault_due: fault.apply / record_inject                                     │
  │                                                                                  │
  │   OBSERVATION (runs for EVERY node — agent and non-agent):                        │
  │     scope   = observer.begin_scope(adapter, node_seq, idx_step) # DetectionSink  │
  │     pre     = observer.inspect(adapter, 'pre')          # 462/650 (boundary)     │
  │     result  = observer.run(callable, adapter, *a, **k)  # inner probes fire      │
  │                 # node-level watchdog → 504 ; escaped exc → classify             │
  │     post    = observer.inspect(adapter, 'post')         # 460/601/640 (boundary) │
  │     inner   = observer.end_scope(scope)                 # inner-call Detections   │
  │     verdict = observer.first_fault(pre, inner, post)    # first fault wins → §5.1 │
  │                                                                                  │
  │   RECOVERY (agent nodes only — snapshot exists; restart machinery is agent-wise):│
  │     if is_agent and verdict.is_restartable and self.restart: -> _recover         │
  │     else: log / record_detection / surface (raise on escaped exception)          │
  └────────────────────────────────────────────────────────────────────────────────────┘

  ┌─ _recover helper (existing restart flow, shared) ──────────────────────┐
  │   checkpoint.restore / adapter.set_input(restart.get_context()) / rerun │
  └─────────────────────────────────────────────────────────────────────────┘
```

**Universal observation.** When the observer is set, *no node returns without
being observed* — agent and non-agent alike flow through the observation phase.
`is_agent` no longer gates whether we observe; it only gates the `idx_step`
counter (injection sampling) and injection itself. Two scoping rules keep cost
and attribution clean:

- **`node_seq` vs `idx_step`.** `idx_step` advances on agent steps only (so the
  sampled injection step still lands correctly and a detection can be correlated
  with the injection on the same `idx_step`). A separate `node_seq` advances on
  *every* node so each scope — including non-agent ones — has a unique ordinal.
- **Snapshot / recover agent nodes only.** `checkpoint.save` /
  `restart.set_context` run only for agent nodes, so per-node checkpoint cost is
  unchanged from today, and a restartable verdict triggers `_recover` only where
  a fresh snapshot exists. Non-agent nodes are still fully *observed* (detect +
  log + OTel); a restartable verdict there is logged/surfaced rather than driving
  the agent-oriented restart machinery.

When `self.observer is None`, the wrapper behaves exactly as today
(injection-only, non-agent nodes pass straight through). When `self.fault is
None` but `self.observer` is set, you get a pure detection+recovery layer with no
injection — useful in production. The phases compose: you can inject a fault
*and* observe whether it was detected (closing the FTO test loop — "did my
detector catch the fault I injected?").

### 3.4 Node execution, with and without the observer

The box-art above shows the *phases* of the wrapper. The two sequence diagrams
below show the *lifetime of a single node's execution* in each mode — and, in
particular, what is and is not visible while the inner agentic loop runs. The
crux is §1.1: the inner loop swallows LLM/tool faults into messages, so without
the observer the only thing the Manager can react to is a fault it injected
itself.

**Observer disabled (`self.observer is None`) — today's behavior.** One node in,
one node out. An inner failure is caught by the loop and turned into a message;
the node returns normally and the Manager never sees it.

```mermaid
sequenceDiagram
    participant M as Manager.wrapped
    participant N as node dispatch<br/>(_execute_node / run_with_retry)
    participant L as inner agentic loop
    participant API as LLM / tool calls

    M->>M: make_adapter()
    alt non-agent node
        M->>N: callable(*args)  %% straight pass-through
        N-->>M: result
    else agent node
        M->>M: idx_step += 1
        opt fault.idx_step == idx_step
            M->>M: checkpoint.save / restart.set_context
            M->>N: fault.apply + record_injection
        end
        M->>N: callable(*args)
        N->>L: drive loop
        L->>API: LLM / tool call
        API--xL: raises (timeout, 502, ban…)
        Note over L,API: ❌ caught → turned into a<br/>message; node still returns "ok"
        L-->>N: degraded result
        N-->>M: result (looks successful)
        opt restart configured
            M->>M: _recover() — only for the injected step
        end
    end
```

The blind spot: the `API--xL` failure never reaches `M`. The only thing the
Manager can act on is the fault it injected itself.

**Observer enabled (`self.observer` set) — observe every node.** Inner probes
wrap the same call sites and record a `Detection` into the context-aware
`DetectionSink` **before** the loop swallows the exception (and then re-raise, so
the framework's own handling is byte-for-byte unchanged). The Manager opens a
scope around the node, drains it afterwards, and computes a verdict.

```mermaid
sequenceDiagram
    participant M as Manager.wrapped
    participant O as Observer
    participant S as DetectionSink<br/>(ContextVar scope)
    participant N as node dispatch
    participant L as inner agentic loop
    participant P as inner probe
    participant API as LLM / tool calls

    M->>M: node_seq += 1 ; if agent: idx_step += 1
    opt agent node
        M->>M: checkpoint.save / restart.set_context
    end
    opt fault_due
        M->>N: fault.apply + record_injection
    end

    M->>O: begin_scope(adapter, node_seq, idx_step)
    O->>S: _active_scope.set(_Scope)
    M->>O: inspect(adapter,'pre')  %% 462 / 650 boundary

    M->>O: run(callable, adapter)
    O->>N: callable(*args)  %% watchdog if timeout_s
    N->>L: drive loop
    L->>P: LLM / tool call
    P->>API: call
    API--xP: raises
    P->>S: record(classify_exception → 5xx)
    P-->>L: re-raise (loop swallows as before)
    L-->>N: degraded result
    N-->>O: result
    Note over O: hang past timeout_s → record 504<br/>escaped exc → classify as 'node'

    M->>O: inspect(adapter,'post')  %% 460 / 601 / 640
    M->>O: end_scope(token)
    O->>S: drain detections
    S-->>M: inner detections
    M->>O: first_fault(pre, inner, post) → verdict

    alt verdict restartable & agent & restart
        M->>M: _recover() — restore + rerun
    else abort / log / surface
        M->>M: record_detection (OTel bridge) + log
    end
```

The fix: the `P->>S: record(...)` step captures the fault at the inner call site,
so it survives the loop's `except`. The Manager now acts on a typed ATP verdict
(§5.1) instead of being blind to inner failures.

---

## 4. Files to add / change

| File                              | Change                                                            |
| --------------------------------- | ----------------------------------------------------------------- |
| `src/fto/atp.py`                  | **new** — ATP status registry + recovery-hint vocabulary          |
| `src/fto/observer.py`             | **rewrite** — `Detection`, `DetectionSink`, `Observer`, aggregation|
| `src/fto/adapters/probes/probe.py`     | **new** — `ObservationProbe` base + `install_probes`        |
| `src/fto/adapters/probes/langgraph.py` | **new** — wraps `ChatGrazie._generate`, `BaseTool.invoke`   |
| `src/fto/adapters/probes/chatdev.py`   | **new** — wraps `_invoke_provider`, `ToolManager.execute_tool` |
| `src/fto/config.py`               | add `observer: Observer \| None`, `probes: ... \| None`           |
| `src/fto/manager.py`              | accept `observer`; add observation + recovery wiring              |
| `src/fto/supervisor.py`           | install probes after instrumentation; pass `observer` to `Manager`|
| `src/fto/__init__.py`             | export `Observer`, detectors, probes, `ATPStatus`, `RecoveryHint` |
| `README.md`                       | document the observer layer                                       |

---

## 5. Step-by-step implementation

### Step 1 — ATP vocabulary module (`src/fto/atp.py`)

Single source of truth for codes and recovery semantics. Keep it dependency-free
(pure data) so it can be unit-tested in isolation.

```python
from dataclasses import dataclass, field
from enum import StrEnum, auto


class RecoveryHint(StrEnum):
    CONTINUE = auto()
    ADVANCE = auto()
    REROUTE = auto()
    RETRY_BACKOFF = auto()
    REPAIR_REQUEST = auto()
    VERIFY_OR_REPAIR = auto()
    HUMAN_REVIEW = auto()
    ABORT = auto()


# Hints for which FTO's restart machinery is the right recovery action.
RESTARTABLE_HINTS: frozenset[RecoveryHint] = frozenset({
    RecoveryHint.RETRY_BACKOFF,
    RecoveryHint.REPAIR_REQUEST,
    RecoveryHint.VERIFY_OR_REPAIR,
})


@dataclass(frozen=True)
class ATPStatus:
    code: int
    name: str
    family: str          # '4xx', '5xx', '6xx', ...
    recovery_hint: RecoveryHint
    fault_tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_restartable(self) -> bool:
        return self.recovery_hint in RESTARTABLE_HINTS


def _s(code, name, family, hint, tags=()):
    return ATPStatus(code, name, family, hint, tuple(tags))


# Curated registry — see the tables in §2.1.
STATUS_REGISTRY: dict[int, ATPStatus] = {s.code: s for s in [
    # exception-shaped
    _s(408, 'Caller-side Timeout',            '4xx', RecoveryHint.RETRY_BACKOFF),
    _s(500, 'Internal Agent Runtime Error',   '5xx', RecoveryHint.RETRY_BACKOFF, ('AgentFail:F3.2', 'TRAIL:service-error')),
    _s(502, 'Bad Upstream Response',          '5xx', RecoveryHint.RETRY_BACKOFF, ('TRAIL:service-error',)),
    _s(503, 'Service Unavailable',            '5xx', RecoveryHint.RETRY_BACKOFF, ('AgentFail:F3.2',)),
    _s(504, 'Gateway or Dependency Timeout',  '5xx', RecoveryHint.RETRY_BACKOFF, ('TRAIL:timeout', 'AgentFail:F3.1')),
    _s(507, 'Resource Exhausted',             '5xx', RecoveryHint.RETRY_BACKOFF, ('TRAIL:resource-exhaustion',)),
    _s(508, 'Runtime Loop Detected',          '5xx', RecoveryHint.ABORT,         ('AgentFail:F2.3', 'MAST:FM-1.3')),
    _s(520, 'Transport Fault',                '5xx', RecoveryHint.RETRY_BACKOFF, ('AgentFail:F3.1',)),
    _s(522, 'Sandbox or Environment Failure', '5xx', RecoveryHint.RETRY_BACKOFF, ('TRAIL:environment-setup',)),
    _s(529, 'Provider-side Rate Limit',       '5xx', RecoveryHint.RETRY_BACKOFF, ('TRAIL:rate-limiting',)),
    # context-shaped
    _s(413, 'Context Too Large',              '4xx', RecoveryHint.REPAIR_REQUEST,  ('TRAIL:context-handling',)),
    _s(460, 'Invalid Agent Output Format',    '4xx', RecoveryHint.REPAIR_REQUEST,  ('AgentFail:F1.2', 'MAST:FM-2.2')),
    _s(462, 'Unsafe Input Detected',          '4xx', RecoveryHint.ABORT,           ('OWASP:prompt-injection',)),
    _s(601, 'Hallucinated Content',           '6xx', RecoveryHint.VERIFY_OR_REPAIR,('TRAIL:language-hallucination',)),
    _s(640, 'Context Loss or Amnesia',        '6xx', RecoveryHint.VERIFY_OR_REPAIR,('MAST:FM-1.4',)),
    _s(642, 'Memory Poisoning Propagated',    '6xx', RecoveryHint.ABORT,           ('OWASP:memory-poisoning',)),
    _s(650, 'Prompt Injection Succeeded',     '6xx', RecoveryHint.ABORT,           ('OWASP:prompt-injection',)),
]}


def status(code: int) -> ATPStatus:
    return STATUS_REGISTRY[code]
```

> **Design note:** `RESTARTABLE_HINTS` is the *policy* knob. If you later want
> `human_review` to also trigger a restart-with-escalation, you change one set —
> no detector, probe, or manager code moves.

### Step 2 — Detection record + context-aware sink (`src/fto/observer.py`, part 1)

`Detection` is the unit of evidence. `DetectionSink` is the side channel inner
probes write to; it is keyed to the active node step via a `ContextVar` so it
propagates into worker threads the same way `instrumentation._session_steps`
does (LangGraph copies context into threads; ChatDev's `ParallelExecutor`
re-attaches OTel context). A single run never observes across two nodes at once
on the same logical thread, so a per-scope list is enough.

```python
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

from fto.adapters.node.node import NodeAdapter
from fto.atp import ATPStatus, RecoveryHint, status


@dataclass
class Detection:
    atp: ATPStatus
    site: str                  # 'pre' | 'llm' | 'tool' | 'node' | 'post'
    detail: str = ''
    node_id: str | None = None
    node_seq: int | None = None    # per-node ordinal (every node)
    idx_step: int | None = None    # agent-step index (agents only; injection corr.)
    exception: BaseException | None = None


@dataclass
class _Scope:
    node_id: str
    node_seq: int                  # advances on every observed node
    idx_step: int | None = None    # set for agent nodes only
    detections: list[Detection] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)


# Active scope for the node currently executing on this logical thread.
_active_scope: ContextVar['_Scope | None'] = ContextVar('_fto_obs_scope', default=None)


class DetectionSink:
    """Where inner probes deposit Detections for the node in flight.

    Probes call `record(...)`; the Manager opens a scope around the node call and
    drains it afterwards. No-op when no scope is active (e.g. observer disabled),
    so probes are always safe to call. Every node — agent or not — gets a scope,
    so inner faults in non-agent nodes are captured too.
    """

    @staticmethod
    def begin(node_id: str, node_seq: int, idx_step: int | None = None):
        token = _active_scope.set(_Scope(node_id, node_seq, idx_step))
        return token

    @staticmethod
    def end(token) -> list[Detection]:
        scope = _active_scope.get()
        _active_scope.reset(token)
        return list(scope.detections) if scope else []

    @staticmethod
    def record(detection: Detection) -> None:
        scope = _active_scope.get()
        if scope is None:
            return
        detection.node_id = detection.node_id or scope.node_id
        detection.node_seq = detection.node_seq if detection.node_seq is not None else scope.node_seq
        detection.idx_step = detection.idx_step if detection.idx_step is not None else scope.idx_step
        scope.detections.append(detection)
```

### Step 3 — Exception/payload classifiers shared by probes (`observer.py`, part 2)

The classification tables live next to the sink so both the inner probes and the
node-boundary inspection reuse them.

```python
# Exception type -> ATP code. Order matters; most specific first. Extend with
# framework/provider-specific exception types (grazie, langchain, ollama, ...).
EXCEPTION_MAPPING: list[tuple[type[BaseException], int]] = [
    (TimeoutError, 504),
    (MemoryError, 507),
    (RecursionError, 508),
    (ConnectionError, 520),
    (ModuleNotFoundError, 522),
    # Exception is the catch-all -> 500
]


def classify_exception(exc: BaseException, site: str) -> Detection:
    for exc_type, code in EXCEPTION_MAPPING:
        if isinstance(exc, exc_type):
            return Detection(status(code), site, detail=repr(exc), exception=exc)
    # Heuristics on message text for provider errors that don't have rich types.
    text = str(exc).lower()
    if 'rate limit' in text or '429' in text:
        return Detection(status(529), site, detail=repr(exc), exception=exc)
    if 'unavailable' in text or '503' in text:
        return Detection(status(503), site, detail=repr(exc), exception=exc)
    return Detection(status(500), site, detail=repr(exc), exception=exc)


def classify_tool_result(tool_name: str, result: Any) -> Detection | None:
    """A tool can 'fail' by returning an error string instead of raising — the
    inner loop converts exceptions to "ERROR calling '<tool>' ..." text. Catch
    that shape so a returned-error reads the same as a raised one."""
    text = str(result)
    if text.startswith(f"ERROR calling '{tool_name}'") or 'Traceback (most recent call last)' in text:
        if 'ModuleNotFoundError' in text or 'No module named' in text:
            return Detection(status(522), 'tool', detail=text[:200])
        return Detection(status(502), 'tool', detail=text[:200])
    return None
```

### Step 4 — Inner-call probes (`src/fto/adapters/probes/`)

Probes are the heart of v2. Each framework module installs wrappers around the
**inner** LLM and tool call sites — the same methods `instrumentation.apply_patches`
wraps — composing *over* whatever is currently bound (so OTel instrumentation and
observation stack cleanly). They record into the sink and **re-raise**; they
never change control flow.

`src/fto/adapters/probes/probe.py`:

```python
class ObservationProbe:
    def install(self) -> None: ...
    def uninstall(self) -> None: ...


def install_probes(probes: list[ObservationProbe]) -> None:
    for p in probes:
        p.install()
```

`src/fto/adapters/probes/langgraph.py` — wraps `ChatGrazie._generate` (LLM) and
`langchain_core.tools.BaseTool.invoke` (tools). These are exactly the points the
inner `_run_agentic_loop` drives, so the probe sees the raise *before* the loop's
`except` swallows it into an `AIMessage`/error `ToolMessage`:

```python
from fto.observer import (
    DetectionSink, Detection, classify_exception, classify_tool_result,
)
from fto.atp import status


class LangGraphProbe:
    def __init__(self):
        self._orig = {}

    def install(self):
        from experiments.langgraph.providers import ChatGrazie
        from langchain_core.tools import BaseTool

        gen = ChatGrazie._generate          # may already be the OTel patch
        inv = BaseTool.invoke

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            try:
                result = gen(self, messages, stop=stop, run_manager=run_manager, **kw)
            except BaseException as exc:                       # noqa: BLE001
                DetectionSink.record(classify_exception(exc, 'llm'))
                raise
            # Optional output-shape inspection (460/601) can go here.
            return result

        def invoke(self, input, config=None, **kw):
            tool_name = getattr(self, 'name', 'unknown')
            try:
                result = inv(self, input, config=config, **kw)
            except BaseException as exc:                       # noqa: BLE001
                DetectionSink.record(classify_exception(exc, 'tool'))
                raise
            hit = classify_tool_result(tool_name, result)
            if hit:
                DetectionSink.record(hit)
            return result

        self._orig = {'gen': (ChatGrazie, gen), 'inv': (BaseTool, inv)}
        ChatGrazie._generate = _generate
        BaseTool.invoke = invoke

    def uninstall(self):
        if not self._orig:
            return
        cls_g, gen = self._orig['gen']; cls_g._generate = gen
        cls_i, inv = self._orig['inv']; cls_i.invoke = inv
        self._orig = {}
```

`src/fto/adapters/probes/chatdev.py` — wraps `AgentNodeExecutor._invoke_provider` (LLM)
and `ToolManager.execute_tool` (tool, **async**):

```python
from fto.observer import DetectionSink, classify_exception, classify_tool_result


class ChatDevProbe:
    def __init__(self):
        self._orig = {}

    def install(self):
        from runtime.node.executor.agent_executor import AgentNodeExecutor
        from runtime.node.agent.tool.tool_manager import ToolManager

        inv = AgentNodeExecutor._invoke_provider
        tool = ToolManager.execute_tool

        def _invoke_provider(self, provider, client, conv, timeline, opts, specs, node):
            try:
                return inv(self, provider, client, conv, timeline, opts, specs, node)
            except BaseException as exc:                       # noqa: BLE001
                DetectionSink.record(classify_exception(exc, 'llm'))
                raise

        async def execute_tool(self, tool_name, arguments, tool_config, *, tool_context=None):
            try:
                result = await tool(self, tool_name, arguments, tool_config,
                                    tool_context=tool_context)
            except BaseException as exc:                       # noqa: BLE001
                DetectionSink.record(classify_exception(exc, 'tool'))
                raise
            hit = classify_tool_result(tool_name, result)
            if hit:
                DetectionSink.record(hit)
            return result

        self._orig = {'inv': (AgentNodeExecutor, inv), 'tool': (ToolManager, tool)}
        AgentNodeExecutor._invoke_provider = _invoke_provider
        ToolManager.execute_tool = execute_tool

    def uninstall(self):
        if not self._orig:
            return
        cls_i, inv = self._orig['inv']; cls_i._invoke_provider = inv
        cls_t, tool = self._orig['tool']; cls_t.execute_tool = tool
        self._orig = {}
```

> **Why probes mirror `instrumentation.py` rather than reuse it.** The OTel
> patches *open spans*; the probes *record Detections*. They wrap the same
> methods for the same reason (that's where inner failures happen), but they are
> orthogonal: you can run observation with OTel off, and the probe composes over
> the OTel patch when OTel is on (install order in §Step 7 guarantees this).
> A future consolidation could have the OTel `llm_call`/`tool_call` context
> managers also feed the sink, but keeping them separate keeps observation usable
> without the OTel dependency.

### Step 5 — The `Observer` orchestrator (`observer.py`, part 3)

The observer owns: node-boundary inspection (kept from v1), a node-level watchdog
for hangs that no inner call surfaces, scope lifecycle, and the **aggregation
policy** that turns a step's worth of detections into a single verdict.

```python
import concurrent.futures


class ContextInspector:
    """Boundary-level context-fault detection (the v1 mechanism). Inspects
    adapter state for the marker FTO itself injects, so the test loop can confirm
    detection even when nothing raised."""

    def __init__(self, signals=None, phases=('pre', 'post')):
        from fto.const import DEFAULT_INJECTED_PROMPT
        self.phases = phases
        self.signals = signals or [
            (lambda a: DEFAULT_INJECTED_PROMPT in str(a.last_message or ''), 650),
        ]

    def inspect(self, adapter: NodeAdapter, phase: str) -> Detection | None:
        if phase not in self.phases:
            return None
        for predicate, code in self.signals:
            try:
                if predicate(adapter):
                    return Detection(status(code), phase, detail=f'signal->{code}')
            except Exception:
                continue
        return None


class Observer:
    def __init__(self, inspector: ContextInspector | None = None,
                 timeout_s: float | None = None, logger=None):
        self.inspector = inspector or ContextInspector()
        self.timeout_s = timeout_s
        self.logger = logger

    # -- scope lifecycle (inner probes write into the active scope) --------
    def begin_scope(self, adapter, node_seq, idx_step=None):
        return DetectionSink.begin(adapter.id, node_seq, idx_step)

    def end_scope(self, token) -> list[Detection]:
        return DetectionSink.end(token)

    # -- boundary inspection ----------------------------------------------
    def inspect(self, adapter, phase) -> Detection | None:
        return self.inspector.inspect(adapter, phase)

    # -- node execution under a watchdog ----------------------------------
    def run(self, callable_, adapter, *args, **kwargs):
        """Run the node. On a hang past timeout_s, record a node-level 504.
        Inner exceptions are recorded by probes and (usually) swallowed by the
        framework, so they surface as Detections, not as raises here. An
        exception that *does* escape the node is classified as a 'node' site."""
        if self.timeout_s is None:
            try:
                return callable_(*args, **kwargs)
            except BaseException as exc:                       # noqa: BLE001
                DetectionSink.record(classify_exception(exc, 'node'))
                return None

        # Propagate the active scope (and all other context vars) into the
        # worker thread so inner probes still write to *this* node's scope.
        # ThreadPoolExecutor does NOT copy context automatically; ctx.run does.
        ctx = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(ctx.run, callable_, *args, **kwargs)
            try:
                return fut.result(timeout=self.timeout_s)
            except concurrent.futures.TimeoutError:
                DetectionSink.record(
                    Detection(status(504), 'node',
                              detail=f'node hang > {self.timeout_s}s'))
                return None
            except BaseException as exc:                       # noqa: BLE001
                DetectionSink.record(classify_exception(exc, 'node'))
                return None

    # -- decision policy: FIRST fault detected wins -----------------------
    @staticmethod
    def first_fault(pre, inner, post) -> Detection | None:
        ordered = ([pre] if pre else []) + list(inner) + ([post] if post else [])
        return ordered[0] if ordered else None

    @staticmethod
    def should_restart(detection: Detection) -> bool:
        return detection.atp.is_restartable
```

> **Watchdog context note.** The active scope lives in a `ContextVar`, and a
> bare `ThreadPoolExecutor.submit` would run the node in a fresh context where
> that var is unset — inner probe `record()` calls would silently no-op. Running
> via `contextvars.copy_context().run(...)` carries the scope (and the OTel
> context) into the worker thread; the `_Scope` object is shared by reference, so
> appends made on the worker are visible when the Manager drains on the main
> thread.

### Step 5.1 — Decision policy: first fault detected wins (`observer.py`)

The restart decision is **first fault detected wins** (`Observer.first_fault`):
the earliest detection — pre-execution boundary check, then inner-call detections
in the order the probes recorded them, then the post-execution boundary check —
becomes the verdict, with **no aggregation or thresholding**. The Manager then
acts on that verdict's recovery hint (restartable → restart on an agent node;
`abort` → surface; otherwise log).

> **Trade-off to know.** First-wins means that if a restartable inner fault
> (e.g. `502`) is recorded *before* an `abort`-class fault (e.g. `642` memory
> poisoning) in the same node, the node restarts on the `502` and the `abort`
> is not prioritized this pass (it would re-surface on the restarted run, where
> the pre-check typically catches it first). If you instead want `abort` to take
> precedence regardless of order, scan `ordered` for an `ABORT`-hint detection
> before falling back to `ordered[0]` — a one-line change in `first_fault`.

### Step 6 — Wire the observer into `Manager` (`src/fto/manager.py`)

Three additions: store `observer` and a per-node counter in `__init__`, add the
shared `_recover` helper, and rewrite `make_supervised_callable` with the two
top-level branches. **Observer disabled** is today's behavior byte-for-byte
(non-agent nodes pass through, injection on the matching agent step). **Observer
enabled observes every node** — `is_agent` only gates the `idx_step` counter and
injection, never whether we observe; snapshot/recovery stay agent-only so
checkpoint cost is unchanged.

```python
def __init__(self, fault, restart, edge_suppressor, logger=None,
             checkpoint=None, observer=None):
    self.fault = fault
    self.restart = restart
    self.edge_suppressor = edge_suppressor
    self.logger = logger
    self.checkpoint = checkpoint
    self.observer = observer
    self.idx_step = 0      # agent-step counter (injection sampling)
    self.node_seq = 0      # per-node ordinal (scope attribution; every node)


def _recover(self, callable, adapter, *args, **kwargs):
    """Restore checkpoint, reset input from the configured Restart, re-run."""
    if self.checkpoint:
        self.checkpoint.restore(node_id=adapter.id)
    adapter.set_input(self.restart.get_context())
    self.logger.info(
        f'Restarting {adapter.id} with {type(self.restart).__name__}.',
        node_id=adapter.id,
    )
    return callable(*args, **kwargs)
```

```python
def make_supervised_callable(self, callable, make_adapter):
    def wrapped(*args, **kwargs):
        adapter: NodeAdapter = make_adapter(*args, **kwargs)

        # ============================================================
        # OBSERVER DISABLED -> existing injection-only behavior,
        # byte-for-byte. Non-agent nodes pass straight through.
        # ============================================================
        if self.observer is None:
            if not adapter.is_agent:
                return callable(*args, **kwargs)

            self.idx_step += 1
            if not self.fault or not self.fault.idx_step == self.idx_step:
                return callable(*args, **kwargs)
            if self.fault.applied:
                return callable(*args, **kwargs)

            if self.checkpoint:
                self.checkpoint.save(node_id=adapter.id)
            if self.restart:
                self.restart.set_context(adapter.input)

            self.fault.apply(node=adapter)
            record_injection(
                node_id=adapter.id, mode=self.fault.mode, idx_step=self.idx_step
            )
            self.logger.info(
                f'Fault {self.fault.mode} applied on {adapter.id}. '
                f'New node input: {adapter.input}',
                node_id=adapter.id,
            )

            token = self.edge_suppressor.suppress(*args, **kwargs) if self.restart else None
            faulty_result = callable(*args, **kwargs)
            self.logger.info('Faulty execution completed.', node_id=adapter.id)

            if token is not None:
                self.edge_suppressor.restore(token, *args, **kwargs)
            if hasattr(self.fault, 'disable'):
                self.fault.disable()

            if not self.restart:
                return faulty_result
            return self._recover(callable, adapter, *args, **kwargs)

        # ============================================================
        # OBSERVER ENABLED -> observe EVERY node (agent and non-agent).
        # No node returns without going through the observer. idx_step
        # and injection remain agent-only.
        # ============================================================
        is_agent = adapter.is_agent
        self.node_seq += 1
        if is_agent:
            self.idx_step += 1   # counts agent steps only (injection sampling)

        fault_due = (
            is_agent
            and self.fault is not None
            and self.fault.idx_step == self.idx_step
            and not self.fault.applied
        )

        # Snapshot only agent nodes: recovery machinery is agent-oriented and a
        # checkpoint per non-agent node would be costly. Captures clean state
        # pre-injection so _recover has a valid restore point for this step.
        if is_agent:
            if self.checkpoint:
                self.checkpoint.save(node_id=adapter.id)
            if self.restart:
                self.restart.set_context(adapter.input)

        if fault_due:
            self.fault.apply(node=adapter)
            record_injection(
                node_id=adapter.id, mode=self.fault.mode, idx_step=self.idx_step
            )
            self.logger.info(
                f'Fault {self.fault.mode} applied on {adapter.id}. '
                f'New node input: {adapter.input}',
                node_id=adapter.id,
            )

        # --- observation phase (runs for every node) ---
        scope = self.observer.begin_scope(
            adapter, self.node_seq, self.idx_step if is_agent else None
        )
        pre = self.observer.inspect(adapter, 'pre')          # 462 / 650 (boundary)

        token = self.edge_suppressor.suppress(*args, **kwargs) if self.restart else None
        result = self.observer.run(callable, adapter, *args, **kwargs)  # inner probes fire
        if token is not None:
            self.edge_suppressor.restore(token, *args, **kwargs)
        if hasattr(self.fault, 'disable'):
            self.fault.disable()

        post = self.observer.inspect(adapter, 'post')        # 460 / 601 / 640 (boundary)
        inner = self.observer.end_scope(scope)               # inner-call Detections
        verdict = self.observer.first_fault(pre, inner, post)

        if verdict is None:
            return result                                    # clean run

        self.logger.info(
            f'Observer: ATP {verdict.atp.code} ({verdict.atp.name}) on {adapter.id} '
            f'[{verdict.atp.recovery_hint}] via {verdict.site}; '
            f'{len(inner)} inner detection(s).',
            node_id=adapter.id,
        )
        record_detection(                                    # OTel bridge, §8
            node_id=adapter.id, code=verdict.atp.code,
            recovery_hint=str(verdict.atp.recovery_hint),
            idx_step=self.idx_step if is_agent else None,
        )

        # Recover agent nodes only (a fresh snapshot exists for them). A
        # restartable verdict on a non-agent node is logged/surfaced instead.
        if is_agent and self.observer.should_restart(verdict) and self.restart:
            return self._recover(callable, adapter, *args, **kwargs)

        # abort / human_review / reroute, non-agent, or no restart configured:
        if verdict.exception is not None and not (self.fault and self.fault.raises_on_fault):
            raise verdict.exception
        return result

    return wrapped
```

> **Why `is_agent` no longer gates observation.** In v1 the wrapper returned
> `callable(...)` early for every non-agent node *and* for every agent step that
> wasn't the injection step — so those nodes were never observed. With the
> observer enabled, the only early `return` is the clean-run case *after*
> observation. `is_agent` is now purely an injection concern.

### Step 7 — Install probes + pass the observer through `Supervisor`

Probes must be installed **after** `self.instrumentation()` so they compose over
the OTel patches (the same reason `Supervisor.start` re-captures `original` after
instrumentation). Add `probes` to the config and install them here.

```python
# src/fto/supervisor.py
class Supervisor:
    def __init__(self, fto_config):
        self.manager = Manager(
            fault=fto_config.fault,
            restart=fto_config.restart,
            edge_suppressor=fto_config.edge_suppressor,
            logger=fto_config.logger,
            checkpoint=fto_config.checkpoint,
            observer=fto_config.observer,        # <-- new
        )
        self.checkpoint = fto_config.checkpoint
        self.instrumentation = fto_config.instrumentation
        self.probes = fto_config.probes          # <-- new
        self.logger = fto_config.logger

    def start(self, target, function_name, make_adapter):
        self.logger.info('Started supervisor.')
        if self.instrumentation:
            self.instrumentation()
        if self.observer is not None and self.probes:   # install inner-call probes
            from fto.adapters.probes.probe import install_probes
            install_probes(self.probes)
        original = getattr(target, function_name)
        if self.checkpoint:
            self.checkpoint.save_baseline()
        # Observer set -> observe every node via wrapper 2; else keep wrapper 1.
        if self.observer is not None:
            wrapped = self.manager.make_supervised_callable_2(original, make_adapter)
        else:
            wrapped = self.manager.make_supervised_callable(original, make_adapter)
        setattr(target, function_name, wrapped)
```

`src/fto/config.py`:

```python
from fto.observer import Observer

@dataclass
class FTOConfig:
    fault: Fault | None = None
    restart: Restart | None = None
    edge_suppressor: EdgeSuppressor = field(default_factory=EdgeSuppressor)
    logger: Any = None
    checkpoint: Checkpoint | None = None
    instrumentation: Callable[..., Any] | None = None
    observer: Observer | None = None                 # None == disabled
    probes: list[Any] | None = None                  # inner-call probes
```

The runner's `build_fto_config` then chooses the framework's probe set, e.g.
ChatDev: `probes=[ChatDevProbe()]`, LangGraph: `probes=[LangGraphProbe()]`,
alongside `observer=Observer(timeout_s=...)`.

### Step 8 — OTel bridge for detections (implemented)

`injection_notice.py` bridges injection → the `agent_step` span via a single-slot
notice that the framework consumes *as it opens* that span. A detection *verdict*
is computed **after the node returns** — after that node's `agent_step` span has
closed — so the verdict cannot use the same "stamp-on-open" trick. But each
individual `Detection` is recorded by an inner probe **while the node is still
executing**, at which point that node's `agent_step` span is the live current
span. The bridge stamps detections there, as they happen.

What is implemented:

- `injection_notice.annotate_detection(detection)` — stamps the current span
  (`trace.get_current_span()`) with `fto.observer.atp_code` / `atp_name` /
  `recovery_hint` / `site` attributes and a `fto.fault.detected` event (also
  carrying `fto.fault.idx_step`). No-op when OTel is inactive or no span is
  recording.
- `DetectionSink.record` invokes a registered emitter hook
  (`detection.set_detection_emitter`) for every recorded detection. The hook
  keeps FTO core free of an OpenTelemetry import; it fires *before* any IMMEDIATE
  `KillNode` interruption so the detection is visible even when the node aborts.
- The framework's instrumentation registers `annotate_detection` as the emitter
  in `apply_patches` (only when OTel is enabled).

Because injection is stamped on the same `agent_step` span (`fto.fault.injected`,
code X) and the detection now lands there too (`fto.fault.detected`, ATP code Y),
a single queryable span answers the core FTO research signal: *did the detector
catch the injected fault?*

- `injection_notice.record_detection(...)` / `take_detections()` remain as a
  lightweight in-memory recorder for logging and end-of-run export; the Manager
  still calls `record_detection(...)` right after a verdict (Step 6).

### Step 9 — Export the public API (`src/fto/__init__.py`)

Export `Observer`, `ContextInspector`, `Detection`, `DetectionSink` from
`fto.observer`; `ObservationProbe`, `install_probes`, and the framework probes
from `fto.adapters.probes`; and `ATPStatus`, `RecoveryHint`, `STATUS_REGISTRY`
from `fto.atp`. (The file is currently empty — establish the package surface
here.)

---

## 6. Worked example (production: detect inner faults + recover, no injection)

```python
from pathlib import Path
from fto import FTOConfig, Supervisor
from fto.observer import Observer
from fto.adapters.probes.chatdev import ChatDevProbe
from fto.recovery import RestartAllContext, GitBranchCheckpoint

config = FTOConfig(
    fault=None,                                  # no injection in production
    restart=RestartAllContext(),                 # recovery for restartable hints
    checkpoint=GitBranchCheckpoint(repo_path=Path('.'), run_id='prod-001'),
    observer=Observer(timeout_s=300),            # node-level hang -> ATP 504
    probes=[ChatDevProbe()],                     # observe inner LLM + tool calls
    logger=my_logger,
)

Supervisor(config).start(
    target=GraphExecutor, function_name='_execute_node',
    make_adapter=lambda instance, node: ChatDevNodeAdapter(node),
)
```

Behavior per outcome (note these are now caught *inside* the node):

- An inner tool call raises `ModuleNotFoundError` → probe records ATP **522**
  (`retry_backoff`); the loop swallows it into an error `ToolMessage` as before.
  If it repeats past threshold (or the tool gets banned) → verdict 522/508 →
  restart from checkpoint.
- An inner LLM call raises a rate-limit error → probe records ATP **529**
  (`retry_backoff`).
- The node hangs past 300 s with no inner call returning → watchdog → ATP **504**
  → restart.
- A tool returns an `"ERROR calling '...'"` string (failure without raising) →
  `classify_tool_result` → ATP **502**.
- Injected/observed prompt-injection marker present at the boundary → ATP **650**
  (`abort`) → **no** restart; logged + surfaced as a security event.
- A single inner tool error the agent immediately recovered from → recorded, but
  aggregation returns `continue` → **no** restart (don't fight the loop).

## 7. Worked example (test loop: inject + observe)

```python
config = FTOConfig(
    fault=PromptInjectionFault(idx_step=3),       # inject on the 3rd agent step
    restart=RestartAllContext(),
    observer=Observer(),                          # boundary marker detector on
    probes=[LangGraphProbe()],                    # + inner-call observation
    checkpoint=GitBranchCheckpoint(repo_path=Path('.'), run_id='exp-007'),
)
```

The injection phase corrupts step 3's input; the boundary `ContextInspector`
flags ATP **650**. With the OTel bridge (Step 8) the `agent_step` span carries
both `injected=650` and `detected=650`, giving a per-run detection-rate metric.
If instead you inject an OTel `llm_call` fault (`OTelFault`, which raises inside
`_invoke_provider` / `_generate`), the **inner probe** is what catches it — a
`5xx` recorded against the `llm` site — demonstrating exactly the inner coverage
v1 lacked.

---

## 8. Testing checklist

1. **`atp.py`** — every entry's `is_restartable` matches its hint; `status(code)`
   round-trips; unknown code raises `KeyError`.
2. **classifiers** — `TimeoutError→504`, `MemoryError→507`, `ConnectionError→520`,
   `ModuleNotFoundError→522`, rate-limit text→529, arbitrary `ValueError→500`;
   `classify_tool_result` fires on `"ERROR calling '<tool>'"` and on tracebacks.
3. **`DetectionSink`** — `record` is a no-op with no active scope; `begin`/`end`
   isolate scopes; detections recorded on a worker thread land in the scope that
   was active when the thread was spawned (ContextVar copy semantics).
4. **probes** — with a probe installed, an inner LLM raise and an inner tool raise
   each deposit exactly one `Detection`, and the exception still propagates so the
   framework's swallow/ban logic is byte-for-byte unchanged (golden test against
   an un-probed run's messages).
5. **`Observer.run`** — clean node returns its result; a node hang past
   `timeout_s` records a `504`; an escaped exception records a `node`-site code.
6. **`first_fault`** — returns the earliest detection in order (pre → inner in
   recorded order → post); `None` when there are no detections; a later
   detection never displaces an earlier one.
7. **`Manager` integration** — with `observer=None`, output is identical to today
   (golden test); with observer + restartable verdict on an agent node, `_recover`
   runs once; with `abort` verdict, no restart and the exception/result surfaces;
   `idx_step` increments exactly once per *agent* dispatch regardless of observer
   state.
8. **universal observation** — with the observer enabled, a *non-agent* node and
   an agent node that is *not* the injection step both open a scope and produce a
   verdict (i.e. neither returns `callable(...)` un-observed); a non-agent node
   does **not** trigger `checkpoint.save` or `_recover`; `node_seq` advances on
   every node while `idx_step` advances only on agent nodes.
9. **end-to-end** — run the planner-coder ReAct workflow with a tool patched to
   raise; confirm the observer records the inner fault that the loop would
   otherwise have hidden.

---

## 9. Summary of the recommended architecture

- **One** supervised node wrapper (extend the existing one) with three gated
  phases: injection (existing), observation (new), recovery (existing, now shared
  via `_recover`).
- **Observation is universal.** When the observer is set, *every* node — agent
  and non-agent, injection step or not — flows through the observation phase; no
  node returns un-observed. `is_agent` only gates the `idx_step` counter and
  injection; snapshot/recovery stay agent-only so checkpoint cost is unchanged.
  A per-node `node_seq` gives non-agent scopes a clean ordinal alongside the
  agent-only `idx_step`.
- **Observation is two-level.** Node-boundary inspection (context markers,
  escaped exceptions, hang watchdog) *plus* **inner-call probes** that wrap the
  LLM/tool call sites — the same sites `instrumentation.py` patches — and record
  faults into a context-aware `DetectionSink` **before** `_run_agentic_loop`
  swallows them. The inner level is the fix for v1's blind spot.
- **`fto.atp`** is the typed-outcome backbone: integer codes → recovery hints,
  with `RESTARTABLE_HINTS` as the single policy knob.
- **An aggregation policy** turns a step's detections into one verdict, so the
  observer restarts on node-level/systemic failure but lets the loop's own
  self-recovery handle one-off inner errors.
- **Probes compose over OTel**, installed after `instrumentation()`; observation
  works with OTel off and enriches the same spans when OTel is on.
- Enable/disable via `FTOConfig.observer is not None` (+ `probes`) — zero behavior
  change when unset.

---

### Sources

- [ATP — Agentic Transfer Protocol](https://atp-protocol.com/) (status-code
  registry and recovery-hint vocabulary).
- Inner-loop fault handling: `experiments/langgraph/instances/planner_coder.py`
  (`AgentStrategy._run_agentic_loop`); inner call sites and the patch technique:
  `experiments/{langgraph,chatdev}/instrumentation.py`.
