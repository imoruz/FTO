from threading import Lock

from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode

from llmmas_otel import semconv

_lock = Lock()
_pending: dict | None = None

FAULT_EVENT = 'fto.fault.injected'
ATTR_FAULT_IDX_STEP = 'fto.fault.idx_step'
ATTR_DETECT_ATP_CODE = 'fto.observer.atp_code'
ATTR_DETECT_ATP_NAME = 'fto.observer.atp_name'
ATTR_DETECT_RECOVERY_HINT = 'fto.observer.recovery_hint'
ATTR_DETECT_SITE = 'fto.observer.site'
ATTR_DETECT_DETAIL = 'fto.observer.detail'
DETECT_EVENT = 'fto.fault.detected'


def record_injection(*, node_id: str, mode, idx_step: int) -> None:
    global _pending
    with _lock:
        _pending = {'node_id': node_id, 'mode': str(mode), 'idx_step': idx_step}


# Sites that correspond to an inner span (llm_call / tool_call).
_INNER_SITES = frozenset({'llm', 'tool'})


def annotate_detection(detection) -> None:
    """Stamp an observer ``Detection`` onto the span for the call it describes.
    """
    span = trace.get_current_span()
    if span is None or not span.is_recording():
        return
    atp = detection.atp
    span.set_attribute(ATTR_DETECT_ATP_CODE, atp.code)
    span.set_attribute(ATTR_DETECT_ATP_NAME, atp.name)
    span.set_attribute(ATTR_DETECT_RECOVERY_HINT, str(atp.recovery_hint))
    span.set_attribute(ATTR_DETECT_SITE, detection.site)
    event_attrs = {
        ATTR_DETECT_ATP_CODE: atp.code,
        ATTR_DETECT_ATP_NAME: atp.name,
        ATTR_DETECT_RECOVERY_HINT: str(atp.recovery_hint),
        ATTR_DETECT_SITE: detection.site,
    }
    if detection.idx_step is not None:
        event_attrs[ATTR_FAULT_IDX_STEP] = detection.idx_step
    if detection.detail:
        event_attrs[ATTR_DETECT_DETAIL] = detection.detail[:200]
    span.add_event(DETECT_EVENT, event_attrs)
    # An inner LLM/tool fault marks its own span errored
    if detection.site in _INNER_SITES:
        span.set_status(Status(StatusCode.ERROR, f'ATP {atp.code} ({atp.name})'))
        if detection.exception is not None:
            span.record_exception(detection.exception)

_detections: list[dict] = []


def record_detection(*, node_id: str, code: int, recovery_hint: str,
                     idx_step: int | None = None) -> None:
    with _lock:
        _detections.append({
            'node_id': node_id,
            'code': code,
            'recovery_hint': recovery_hint,
            'idx_step': idx_step,
        })


def take_detections() -> list[dict]:
    """Pop and return all recorded detections
    """
    global _detections
    with _lock:
        out, _detections = _detections, []
        return out


def take_injection(node_id: str | None = None) -> dict | None:
    """Pop and return the pending injection notice
    """
    global _pending
    with _lock:
        if _pending is None:
            return None
        if node_id is not None and _pending['node_id'] != node_id:
            return None
        notice, _pending = _pending, None
        return notice


def annotate_span(span, node_id: str) -> None:
    """Stamp the pending fault-injection note onto a span.
    """
    notice = take_injection(node_id)
    if notice is None or span is None:
        return
    span.set_attribute(semconv.ATTR_FAULT_INJECTED, True)
    span.set_attribute(semconv.ATTR_FAULT_TYPE, notice['mode'])
    span.set_attribute(ATTR_FAULT_IDX_STEP, notice['idx_step'])
    span.add_event(
        FAULT_EVENT,
        {
            semconv.ATTR_FAULT_TYPE: notice['mode'],
            ATTR_FAULT_IDX_STEP: notice['idx_step'],
        },
    )
