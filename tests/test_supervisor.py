from fto.config import FTOConfig
from fto.manager import Manager
from fto.supervisor import Supervisor


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg, node_id=None):
        self.messages.append(msg)


class FakeObserver:
    def __init__(self, timeout_llm=1.5, timeout_tool=2.5):
        self.timeout_llm = timeout_llm
        self.timeout_tool = timeout_tool


class FakeProbe:
    def __init__(self):
        self.install_calls = []

    def install(self, timeout_llm=None, timeout_tool=None):
        self.install_calls.append((timeout_llm, timeout_tool))


class FakeCheckpoint:
    def __init__(self):
        self.baseline_saved = 0

    def save_baseline(self):
        self.baseline_saved += 1


class FakeAdapter:
    def __init__(self, is_agent=False):
        self.is_agent = is_agent
        self.id = 'n1'


class Target:
    def action(self, *args, **kwargs):
        return 'original-result'


def make_config(**overrides):
    kwargs = dict(
        fault=None,
        restart=None,
        logger=FakeLogger(),
        checkpoint=None,
        instrumentation=None,
        observer=None,
        probe=None,
    )
    kwargs.update(overrides)
    return FTOConfig(**kwargs)


class TestSupervisorInit:
    def test_builds_manager_from_config(self):
        fault, restart, edge_suppressor = object(), object(), object()
        checkpoint = FakeCheckpoint()
        observer = FakeObserver()
        config = make_config(
            fault=fault, restart=restart, edge_suppressor=edge_suppressor,
            checkpoint=checkpoint, observer=observer,
        )

        supervisor = Supervisor(config)

        assert isinstance(supervisor.manager, Manager)
        assert supervisor.manager.fault is fault
        assert supervisor.manager.restart is restart
        assert supervisor.manager.edge_suppressor is edge_suppressor
        assert supervisor.manager.checkpoint is checkpoint
        assert supervisor.manager.observer is observer

    def test_stores_config_fields_on_self(self):
        checkpoint = FakeCheckpoint()
        observer = FakeObserver()
        probe = FakeProbe()
        logger = FakeLogger()
        def instrumentation():
            return None
        config = make_config(
            checkpoint=checkpoint, observer=observer, probe=probe,
            logger=logger, instrumentation=instrumentation,
        )

        supervisor = Supervisor(config)

        assert supervisor.checkpoint is checkpoint
        assert supervisor.observer is observer
        assert supervisor.probe is probe
        assert supervisor.logger is logger
        assert supervisor.instrumentation is instrumentation


class TestSupervisorStart:
    def test_logs_start_message(self):
        supervisor = Supervisor(make_config())
        target = Target()

        supervisor.start(target, 'action', lambda *a, **kw: FakeAdapter())

        assert 'Started supervisor.' in supervisor.logger.messages

    def test_installs_probe_with_observer_timeouts_when_both_present(self):
        observer = FakeObserver(timeout_llm=3.0, timeout_tool=4.0)
        probe = FakeProbe()
        supervisor = Supervisor(make_config(observer=observer, probe=probe))
        target = Target()

        supervisor.start(target, 'action', lambda *a, **kw: FakeAdapter())

        assert probe.install_calls == [(3.0, 4.0)]

    def test_does_not_install_probe_when_observer_missing(self):
        probe = FakeProbe()
        supervisor = Supervisor(make_config(observer=None, probe=probe))
        target = Target()

        supervisor.start(target, 'action', lambda *a, **kw: FakeAdapter())

        assert probe.install_calls == []

    def test_does_not_install_probe_when_probe_missing(self):
        observer = FakeObserver()
        supervisor = Supervisor(make_config(observer=observer, probe=None))
        target = Target()

        supervisor.start(target, 'action', lambda *a, **kw: FakeAdapter())  # should not raise

    def test_calls_instrumentation_when_present(self):
        calls = []
        supervisor = Supervisor(make_config(instrumentation=lambda: calls.append(1)))
        target = Target()

        supervisor.start(target, 'action', lambda *a, **kw: FakeAdapter())

        assert calls == [1]

    def test_skips_instrumentation_when_absent(self):
        supervisor = Supervisor(make_config(instrumentation=None))
        target = Target()

        supervisor.start(target, 'action', lambda *a, **kw: FakeAdapter())  # should not raise

    def test_saves_baseline_when_checkpoint_present(self):
        checkpoint = FakeCheckpoint()
        supervisor = Supervisor(make_config(checkpoint=checkpoint))
        target = Target()

        supervisor.start(target, 'action', lambda *a, **kw: FakeAdapter())

        assert checkpoint.baseline_saved == 1

    def test_skips_baseline_when_no_checkpoint(self):
        supervisor = Supervisor(make_config(checkpoint=None))
        target = Target()

        supervisor.start(target, 'action', lambda *a, **kw: FakeAdapter())  # should not raise

    def test_wraps_target_method_with_supervised_callable(self):
        supervisor = Supervisor(make_config())
        target = Target()

        supervisor.start(target, 'action', lambda *a, **kw: FakeAdapter(is_agent=False))

        assert target.action is not Target.action
        assert target.action() == 'original-result'

    def test_captures_original_after_instrumentation_has_patched_target(self):
        """`original` must be re-read from `target` after instrumentation() runs,
        so it wraps whatever instrumentation installed, not the pre-patch function."""
        def patch_target():
            target.action = lambda *a, **kw: 'patched-result'

        target = Target()
        supervisor = Supervisor(make_config(instrumentation=patch_target))

        supervisor.start(target, 'action', lambda *a, **kw: FakeAdapter(is_agent=False))

        assert target.action() == 'patched-result'

    def test_baseline_saved_before_wrapping_but_after_instrumentation(self):
        order = []
        checkpoint = FakeCheckpoint()
        checkpoint.save_baseline = lambda: order.append('baseline')
        supervisor = Supervisor(make_config(
            checkpoint=checkpoint,
            instrumentation=lambda: order.append('instrumentation'),
        ))
        target = Target()

        supervisor.start(target, 'action', lambda *a, **kw: FakeAdapter())

        assert order == ['instrumentation', 'baseline']
