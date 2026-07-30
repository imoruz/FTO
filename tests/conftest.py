"""Shared test setup.

`aegis_mas.aegis_core` pulls in a heavy transitive dependency chain (openai,
requests, ...) that isn't needed to exercise fto's own adapter code. We stub
it once, before any `fto` module is imported, so `from aegis_mas.aegis_core
import AgentContext` (used by fto.detection.atp and fto.adapters.node.*)
resolves without requiring the real package tree to be installed.
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import pytest


def _install_aegis_core_stub() -> None:
    if 'aegis_mas.aegis_core' in sys.modules:
        return

    import aegis_mas  # real top-level package; importable on its own

    module = types.ModuleType('aegis_mas.aegis_core')

    class FMErrorType:
        FM_2_2 = 'FM_2_2'
        FM_2_6 = 'FM_2_6'

    class AgentContext:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __eq__(self, other):
            return isinstance(other, AgentContext) and self.__dict__ == other.__dict__

    class FMMaliciousFactory:
        def __init__(self, *args, **kwargs):
            pass

        def inject_prompt(self, *args, **kwargs):
            raise NotImplementedError('stubbed for adapter tests; not exercised')

    module.FMErrorType = FMErrorType
    module.AgentContext = AgentContext
    module.FMMaliciousFactory = FMMaliciousFactory

    sys.modules['aegis_mas.aegis_core'] = module
    aegis_mas.aegis_core = module


_install_aegis_core_stub()


def register_fake_module(monkeypatch: pytest.MonkeyPatch, dotted_name: str, module: types.ModuleType | None = None) -> types.ModuleType:
    """Install a synthetic module at `dotted_name` in sys.modules for the duration of a test.

    Creates any missing parent packages as empty synthetic modules too, so
    `from a.b.c import D` style imports resolve without touching the filesystem.
    Reverted automatically by monkeypatch at test teardown.
    """
    if module is None:
        module = types.ModuleType(dotted_name)

    monkeypatch.setitem(sys.modules, dotted_name, module)

    if '.' in dotted_name:
        parent_name, attr = dotted_name.rsplit('.', 1)
        parent = sys.modules.get(parent_name)
        if parent is None:
            parent = register_fake_module(monkeypatch, parent_name)
        monkeypatch.setattr(parent, attr, module, raising=False)

    return module


@pytest.fixture
def fake_module(monkeypatch):
    def _make(dotted_name, **attrs):
        module = types.ModuleType(dotted_name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return register_fake_module(monkeypatch, dotted_name, module)

    return _make


@contextmanager
def _detection_scope_impl():
    from fto.detection.detection import DetectionSink

    token = DetectionSink.begin('test-node', 0)
    ended = {}

    def _end():
        if not ended:
            ended['detections'] = DetectionSink.end(token)
        return ended['detections']

    try:
        yield _end
    finally:
        if not ended:
            DetectionSink.end(token)


@pytest.fixture
def detection_scope():
    """Open a DetectionSink scope and hand back a way to read what got recorded.

    Backed by a contextvars.Token, which is only valid within the Context it
    was created in. pytest-asyncio runs each async test's body in its own
    Task (a forked Context), so a fixture-managed token created before that
    Task starts can't be reset from inside the test. Async tests should use
    `detection_scope_cm` directly in their body instead, so begin/end happen
    in the same Context as the test.
    """
    with _detection_scope_impl() as end:
        yield end


detection_scope_cm = _detection_scope_impl
