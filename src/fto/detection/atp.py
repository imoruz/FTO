from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Callable

from aegis_mas.aegis_core import FMErrorType


class RecoveryHint(StrEnum):
    CONTINUE = auto()
    ADVANCE = auto()
    REROUTE = auto()
    RETRY_BACKOFF = auto()
    REPAIR_REQUEST = auto()
    VERIFY_OR_REPAIR = auto()
    HUMAN_REVIEW = auto()
    ABORT = auto()

#TODO: adapt this to be dynamic from observer's init
# Hints for which FTO's restart machinery is the right recovery action.
RESTARTABLE_HINTS = set({
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

    def is_restartable(self) -> bool:
        return self.recovery_hint in RESTARTABLE_HINTS


def _s(code, name, family, hint, tags=()):
    return ATPStatus(code, name, family, hint, tuple(tags))

# AEGIS injected failure mode -> ATP code.
AEGIS_FM_TO_ATP: dict[FMErrorType, int] = {
    FMErrorType.FM_2_2: 622,
    FMErrorType.FM_2_6: 605,
}


# OTel (llmmas_otel) fault spec `action.type` -> ATP code.
OTEL_ACTION_TO_ATP: dict[str, int] = {
    # ---- A2A (agent-to-agent transport) faults ----
    'a2a.drop':                520,
    'a2a.delay':               504,
    'a2a.truncate':            502,
    # ---- Tool faults ----
    'tool.delay':              504,
    'tool.not_installed':      404,
    'tool.timeout':            504,
    'tool.malformed_response': 502,
    # ---- LLM faults ----
    'llm.delay':               504,
    'llm.rate_limit':          529,
    'llm.timeout':             504,
    'llm.network_error':       520,
    'llm.malformed_response':  502,
}


def otel_action_site(action_type: str | None) -> str:
    """Map an OTel action.type to the Detection `site` it would be observed at."""
    prefix = action_type.split('.', 1)[0] if action_type else ''
    if prefix == 'tool':
        return 'tool'
    if prefix == 'llm':
        return 'llm'
    return 'node'  # a2a transport faults and anything unmapped


# Exception type -> ATP code, most specific first. 
EXCEPTION_MAPPING: list[tuple[type[BaseException], int]] = [
    (TimeoutError, 504),
    (MemoryError, 507),
    (RecursionError, 508),
    (ConnectionError, 520),
    (ModuleNotFoundError, 522),
    (FileNotFoundError, 404),
]

STATUS_REGISTRY: dict[int, ATPStatus] = {s.code: s for s in [
    _s(401, 'Authentication Required',        '4xx', RecoveryHint.REPAIR_REQUEST),
    _s(404, 'Agent, Tool or Resource Not Found', '4xx', RecoveryHint.REPAIR_REQUEST, ('TRAIL:resource-not-found')),
    _s(408, 'Caller-side Timeout',            '4xx', RecoveryHint.RETRY_BACKOFF),
    _s(500, 'Internal Agent Runtime Error',   '5xx', RecoveryHint.RETRY_BACKOFF, ('AgentFail:F3.2', 'TRAIL:service-error')),
    _s(413, 'Context Too Large',              '4xx', RecoveryHint.REPAIR_REQUEST,  ('TRAIL:context-handling',)),
    _s(460, 'Invalid Agent Output Format',    '4xx', RecoveryHint.REPAIR_REQUEST,  ('AgentFail:F1.2', 'MAST:FM-2.2')),
    _s(462, 'Unsafe Input Detected',          '4xx', RecoveryHint.ABORT,           ('OWASP:prompt-injection',)),
    _s(502, 'Bad Upstream Response',          '5xx', RecoveryHint.RETRY_BACKOFF, ('TRAIL:service-error',)),
    _s(503, 'Service Unavailable',            '5xx', RecoveryHint.RETRY_BACKOFF, ('AgentFail:F3.2',)),
    _s(504, 'Gateway or Dependency Timeout',  '5xx', RecoveryHint.RETRY_BACKOFF, ('TRAIL:timeout', 'AgentFail:F3.1')),
    _s(507, 'Resource Exhausted',             '5xx', RecoveryHint.RETRY_BACKOFF, ('TRAIL:resource-exhaustion',)),
    _s(508, 'Runtime Loop Detected',          '5xx', RecoveryHint.ABORT,         ('AgentFail:F2.3', 'MAST:FM-1.3')),
    _s(520, 'Transport Fault',                '5xx', RecoveryHint.RETRY_BACKOFF, ('AgentFail:F3.1',)),
    _s(522, 'Sandbox or Environment Failure', '5xx', RecoveryHint.RETRY_BACKOFF, ('TRAIL:environment-setup',)),
    _s(529, 'Provider-side Rate Limit',       '5xx', RecoveryHint.RETRY_BACKOFF, ('TRAIL:rate-limiting',)),
    _s(601, 'Hallucinated Content',           '6xx', RecoveryHint.VERIFY_OR_REPAIR,('TRAIL:language-hallucination',)),
    _s(605, 'Inconsistent Reasoning',         '6xx', RecoveryHint.VERIFY_OR_REPAIR,('MAST:FM-2.6',)),
    _s(622, 'Ambiguous Agent Request',        '6xx', RecoveryHint.VERIFY_OR_REPAIR,('MAST:FM-2.2')),
    _s(626, 'Cross-agent Interface Mismatch'  '6xx', RecoveryHint.VERIFY_OR_REPAIR,('AgentFail:F2.7')),
    _s(640, 'Context Loss or Amnesia',        '6xx', RecoveryHint.VERIFY_OR_REPAIR,('MAST:FM-1.4', 'TRAIL:context-handling')),
    _s(642, 'Memory Poisoning Propagated',    '6xx', RecoveryHint.ABORT,           ('OWASP:memory-poisoning',)),
    _s(650, 'Prompt Injection Succeeded',     '6xx', RecoveryHint.ABORT,           ('OWASP:prompt-injection',)),
]}


def status(code: int) -> ATPStatus:
    return STATUS_REGISTRY[code]
