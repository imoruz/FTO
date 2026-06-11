# Provider/framework-specific classifiers, registered by downstream packages
from typing import Any, Callable

from fto.detection.atp import EXCEPTION_MAPPING, status
from fto.detection.detection import Detection


def register_exception_classifier(
    fn: Callable[[BaseException, str], 'Detection | None'],
) -> None:
    """Register a provider-specific exception classifier.
    """
    _EXTRA_CLASSIFIERS.append(fn)

_EXTRA_CLASSIFIERS: list[Callable[[BaseException, str], 'Detection | None']] = []


def classify_exception(exc: BaseException, site: str) -> Detection:
    for exc_type, code in EXCEPTION_MAPPING:
        if isinstance(exc, exc_type):
            return Detection(status(code), site, detail=repr(exc), exception=exc)
    for classifier in _EXTRA_CLASSIFIERS:
        detection = classifier(exc, site)
        if detection is not None:
            return detection
    if site == 'tool':
        if isinstance(exc, TypeError):
            return Detection(status(460), site, detail=repr(exc), exception=exc)
        if isinstance(exc, ValueError):
              return Detection(status(601), site, detail=repr(exc), exception=exc)
    text = str(exc).lower()
    if 'rate limit' in text or '429' in text:
        return Detection(status(529), site, detail=repr(exc), exception=exc)
    if 'unavailable' in text or '503' in text:
        return Detection(status(503), site, detail=repr(exc), exception=exc)
    return Detection(status(500), site, detail=repr(exc), exception=exc)


def classify_tool_result(tool_name: str, result: Any) -> Detection | None:
    # TODO: remove
    text = str(result)
    if text.startswith(f"ERROR calling '{tool_name}'") or \
            'Traceback (most recent call last)' in text:
        if 'ModuleNotFoundError' in text or 'No module named' in text:
            return Detection(status(522), 'tool', detail=text[:200])
        return Detection(status(500), 'tool', detail=text[:200])
    return None
