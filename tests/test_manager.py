import pytest

from fto.adapters.edge import MethodSwapEdgeSuppressor
from fto.detection.atp import status
from fto.detection.detection import Detection
from fto.instrumentation import take_detections, take_injection
from fto.manager import Manager
from fto.recovery.compression import CompressionResult, ContextCompressor
from fto.recovery.restart import Restart, RestartRefinedContext


@pytest.fixture(autouse=True)
def _clean_instrumentation_globals():
    """record_injection/record_detection stash into module-level globals;
    drain them before and after each test so tests don't leak into each other."""
    take_injection()
    take_detections()
    yield
    take_injection()
    take_detections()


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg, node_id=None):
        self.messages.append((msg, node_id))


class FakeAdapter:
    def __init__(self, id='node-1', is_agent=True, input='initial-input'):
        self.id = id
        self.is_agent = is_agent
        self.input = input
        self.set_input_calls = []

    def set_input(self, value):
        self.set_input_calls.append(value)
        self.input = value


class FakeFault:
    def __init__(self, idx_step, mode='FM_TEST', node_id=None, with_disable=True):
        self.idx_step = idx_step
        self.mode = mode
        self.node_id = node_id
        self.applied = False
        self.apply_calls = []
        self.disable_calls = 0
        if with_disable:
            self.disable = self._disable

    def apply(self, node):
        self.applied = True
        self.apply_calls.append(node)

    def _disable(self):
        self.disable_calls += 1


class FakeRestart(Restart):
    def __init__(self, restart_count=1, contexts=None):
        super().__init__(restart_count=restart_count)
        # sequence of values returned by successive get_context() calls
        self._contexts = list(contexts) if contexts is not None else ['restored-context']

    def get_context(self):
        if len(self._contexts) > 1:
            return self._contexts.pop(0)
        return self._contexts[0]


class FakeEdgeSuppressor:
    def __init__(self):
        self.suppress_calls = []
        self.restore_calls = []
        self.reset_calls = []

    def suppress(self, *args, **kwargs):
        self.suppress_calls.append((args, kwargs))
        return 'token'

    def restore(self, token, *args, **kwargs):
        self.restore_calls.append((token, args, kwargs))

    def reset(self, token, *args, **kwargs):
        self.reset_calls.append((token, args, kwargs))


class FakeCheckpoint:
    def __init__(self):
        self.saved = []
        self.restored = []
        self.baseline_saved = 0

    def save(self, node_id):
        self.saved.append(node_id)

    def restore(self, node_id):
        self.restored.append(node_id)

    def save_baseline(self):
        self.baseline_saved += 1


def make_detection(code=500, site='node'):
    return Detection(status(code), site)


class FakeObserver:
    def __init__(self, detections_by_call=None, restart_decision=True):
        self.timeout_llm = 1.0
        self.timeout_tool = 2.0
        self.record_injected_faults = False
        self.begin_scope_calls = []
        self.end_scope_calls = []
        self.record_injected_fault_calls = []
        self.run_calls = []
        # list of detection-lists returned by successive end_scope calls
        self._detections_by_call = list(detections_by_call) if detections_by_call is not None else [[]]
        self._restart_decision = restart_decision

    def begin_scope(self, adapter, node_seq, idx_step=None, can_restart=False):
        token = ('scope', node_seq, idx_step, can_restart)
        self.begin_scope_calls.append(token)
        return token

    def run(self, callable, adapter, *args, **kwargs):
        self.run_calls.append((adapter, args, kwargs))
        return callable(*args, **kwargs)

    def end_scope(self, token):
        self.end_scope_calls.append(token)
        if len(self._detections_by_call) > 1:
            return self._detections_by_call.pop(0)
        return self._detections_by_call[0]

    def should_restart(self, detections):
        return self._restart_decision

    def _record_injected_fault(self, idx_step, node_id, fault):
        self.record_injected_fault_calls.append((idx_step, node_id, fault))


def make_manager(**overrides):
    kwargs = dict(
        fault=None,
        restart=None,
        edge_suppressor=FakeEdgeSuppressor(),
        logger=FakeLogger(),
        checkpoint=None,
        observer=None,
    )
    kwargs.update(overrides)
    return Manager(**kwargs)


class TestSnapshot:
    def test_saves_checkpoint_and_sets_restart_context_when_both_configured(self):
        checkpoint = FakeCheckpoint()
        restart = FakeRestart()
        manager = make_manager(checkpoint=checkpoint, restart=restart)
        adapter = FakeAdapter(id='n1', input='the-input')

        manager.snapshot(adapter)

        assert checkpoint.saved == ['n1']
        assert restart.context == 'the-input'

    def test_hands_the_adapter_to_the_restart_alongside_the_context(self):
        # Refined-context restart reads and rebuilds the context through it.
        restart = FakeRestart()
        manager = make_manager(restart=restart)
        adapter = FakeAdapter(id='n1', input='the-input')

        manager.snapshot(adapter)

        assert restart.adapter is adapter

    def test_noop_when_neither_checkpoint_nor_restart_configured(self):
        manager = make_manager()
        adapter = FakeAdapter()

        manager.snapshot(adapter)  # should not raise


class TestApplyFault:
    def test_applies_fault_logs_and_records_injection(self):
        fault = FakeFault(idx_step=1, mode='FM_2_2')
        manager = make_manager(fault=fault)
        manager.idx_step = 1
        adapter = FakeAdapter(id='n1')

        manager.apply_fault(adapter)

        assert fault.apply_calls == [adapter]
        assert manager.logger.messages
        notice = take_injection('n1')
        assert notice == {'node_id': 'n1', 'mode': 'FM_2_2', 'idx_step': 1}


class TestFaultDue:
    def test_false_when_no_fault_configured(self):
        manager = make_manager()
        assert not manager._fault_due()

    def test_true_when_idx_step_matches_and_not_applied(self):
        manager = make_manager(fault=FakeFault(idx_step=2))
        manager.idx_step = 2
        assert manager._fault_due() is True

    def test_false_when_idx_step_does_not_match(self):
        manager = make_manager(fault=FakeFault(idx_step=2))
        manager.idx_step = 3
        assert manager._fault_due() is False

    def test_false_when_already_applied(self):
        fault = FakeFault(idx_step=2)
        fault.applied = True
        manager = make_manager(fault=fault)
        manager.idx_step = 2
        assert manager._fault_due() is False


class TestRun:
    def test_calls_callable_and_returns_result(self):
        manager = make_manager()
        assert manager._run(lambda: 'result', fault_applied=False) == 'result'

    def test_disables_fault_when_fault_applied_and_disable_available(self):
        fault = FakeFault(idx_step=1)
        manager = make_manager(fault=fault)
        manager._run(lambda: 'result', fault_applied=True)
        assert fault.disable_calls == 1

    def test_does_not_disable_fault_when_fault_applied_is_false(self):
        fault = FakeFault(idx_step=1)
        manager = make_manager(fault=fault)
        manager._run(lambda: 'result', fault_applied=False)
        assert fault.disable_calls == 0

    def test_does_not_error_when_fault_has_no_disable(self):
        fault = FakeFault(idx_step=1, with_disable=False)
        manager = make_manager(fault=fault)
        result = manager._run(lambda: 'result', fault_applied=True)
        assert result == 'result'


class TestSuppressRestoreEdge:
    def test_suppress_edge_noop_without_restart(self):
        edge_suppressor = FakeEdgeSuppressor()
        manager = make_manager(edge_suppressor=edge_suppressor)
        assert manager._suppress_edge('a', b=1) is None
        assert edge_suppressor.suppress_calls == []

    def test_suppress_edge_delegates_when_restart_configured(self):
        edge_suppressor = FakeEdgeSuppressor()
        manager = make_manager(edge_suppressor=edge_suppressor, restart=FakeRestart())
        token = manager._suppress_edge('a', b=1)
        assert token == 'token'
        assert edge_suppressor.suppress_calls == [(('a',), {'b': 1})]

    def test_restore_edge_noop_when_token_is_none(self):
        edge_suppressor = FakeEdgeSuppressor()
        manager = make_manager(edge_suppressor=edge_suppressor)
        manager._restore_edge(None, 'a', b=1)
        assert edge_suppressor.restore_calls == []

    def test_restore_edge_delegates_when_token_present(self):
        edge_suppressor = FakeEdgeSuppressor()
        manager = make_manager(edge_suppressor=edge_suppressor)
        manager._restore_edge('token', 'a', b=1)
        assert edge_suppressor.restore_calls == [('token', ('a',), {'b': 1})]


class TestMakeSupervisedCallable:
    def test_non_agent_node_calls_original_callable_directly(self):
        manager = make_manager()
        calls = []

        def original(*a, **kw):
            calls.append((a, kw))
            return 'ok'

        def make_adapter(*a, **kw):
            return FakeAdapter(is_agent=False)

        wrapped = manager.make_supervised_callable(original, make_adapter)
        result = wrapped('x', y=1)

        assert result == 'ok'
        assert calls == [(('x',), {'y': 1})]
        assert manager.idx_step == 0

    def test_agent_node_without_observer_uses_injection_only_path(self):
        manager = make_manager()

        def original(*a, **kw):
            return 'ok'

        def make_adapter(*a, **kw):
            return FakeAdapter(is_agent=True)

        wrapped = manager.make_supervised_callable(original, make_adapter)
        result = wrapped()

        assert result == 'ok'
        assert manager.idx_step == 1  # _run_injection_only always bumps idx_step

    def test_agent_node_with_observer_uses_observed_path(self):
        observer = FakeObserver()
        manager = make_manager(observer=observer)

        def original(*a, **kw):
            return 'ok'

        def make_adapter(*a, **kw):
            return FakeAdapter(is_agent=True)

        wrapped = manager.make_supervised_callable(original, make_adapter)
        result = wrapped()

        assert result == 'ok'
        assert observer.begin_scope_calls  # _run_observed went through _observe


class TestRunInjectionOnly:
    def test_no_fault_due_just_calls_callable(self):
        manager = make_manager()
        adapter = FakeAdapter()
        calls = []

        def callable_():
            calls.append(1)
            return 'ok'

        result = manager._run_injection_only(callable_, adapter)

        assert result == 'ok'
        assert calls == [1]
        assert manager.idx_step == 1

    def test_fault_due_without_restart_runs_faulty_execution_once(self):
        fault = FakeFault(idx_step=1)
        checkpoint = FakeCheckpoint()
        edge_suppressor = FakeEdgeSuppressor()
        manager = make_manager(fault=fault, checkpoint=checkpoint, edge_suppressor=edge_suppressor)
        adapter = FakeAdapter(id='n1')
        calls = []

        def callable_():
            calls.append(1)
            return 'faulty-result'

        result = manager._run_injection_only(callable_, adapter)

        assert result == 'faulty-result'
        assert calls == [1]
        assert checkpoint.saved == ['n1']
        assert fault.applied is True
        # no restart configured -> edge never suppressed
        assert edge_suppressor.suppress_calls == []

    def test_fault_due_with_restart_restores_and_reruns(self):
        fault = FakeFault(idx_step=1)
        restart = FakeRestart(contexts=['restored'])
        checkpoint = FakeCheckpoint()
        edge_suppressor = FakeEdgeSuppressor()
        manager = make_manager(
            fault=fault, restart=restart, checkpoint=checkpoint, edge_suppressor=edge_suppressor,
        )
        adapter = FakeAdapter(id='n1')
        call_log = []

        def callable_():
            call_log.append(1)
            return f'result-{len(call_log)}'

        result = manager._run_injection_only(callable_, adapter)

        # first call = faulty run, second call = restart re-run (no observer -> single attempt)
        assert call_log == [1, 1]
        assert result == 'result-2'
        assert checkpoint.restored == ['n1']
        assert adapter.set_input_calls == ['restored']
        assert edge_suppressor.suppress_calls  # edge suppressed around the faulty run
        assert edge_suppressor.restore_calls


class TestRunObserved:
    def test_no_fault_no_detections_no_restart(self):
        observer = FakeObserver(detections_by_call=[[]])
        manager = make_manager(observer=observer)
        adapter = FakeAdapter(id='n1')

        def callable_():
            return 'ok'

        result = manager._run_observed(callable_, adapter)

        assert result == 'ok'
        assert manager.idx_step == 1

    def test_detections_trigger_restart_when_configured(self):
        detections = [make_detection(500)]
        observer = FakeObserver(detections_by_call=[detections, []], restart_decision=True)
        restart = FakeRestart(contexts=['restored'])
        checkpoint = FakeCheckpoint()
        manager = make_manager(observer=observer, restart=restart, checkpoint=checkpoint)
        adapter = FakeAdapter(id='n1')

        def callable_():
            return 'ok'

        result = manager._run_observed(callable_, adapter)

        assert checkpoint.restored == ['n1']
        assert adapter.set_input_calls == ['restored']
        # second observed pass returned no detections -> should_restart loop stops, result kept
        assert result == 'ok'

    def test_detections_without_restart_configured_do_not_restart(self):
        detections = [make_detection(500)]
        observer = FakeObserver(detections_by_call=[detections])
        manager = make_manager(observer=observer)  # no restart configured
        adapter = FakeAdapter(id='n1')

        result = manager._run_observed(lambda: 'ok', adapter)

        assert result == 'ok'
        assert adapter.set_input_calls == []

    def test_fault_applied_forces_restart_even_without_detections(self):
        fault = FakeFault(idx_step=1)
        observer = FakeObserver(detections_by_call=[[], []])
        restart = FakeRestart(contexts=['restored'])
        manager = make_manager(fault=fault, observer=observer, restart=restart)
        manager.idx_step = 0  # will become 1 on this call, matching fault.idx_step
        adapter = FakeAdapter(id='n1')

        manager._run_observed(lambda: 'ok', adapter)

        assert fault.applied is True
        assert adapter.set_input_calls == ['restored']


class TestObserve:
    def test_begins_and_ends_scope_and_returns_result_and_detections(self):
        detections = [make_detection(504, site='tool')]
        observer = FakeObserver(detections_by_call=[detections])
        manager = make_manager(observer=observer)
        adapter = FakeAdapter(id='n1', is_agent=True)

        result, faults = manager._observe(lambda: 'ok', adapter, fault_applied=False)

        assert result == 'ok'
        assert faults == detections
        assert manager.node_seq == 1
        # idx_step defaults to 0 here since we called _observe directly, bypassing
        # the idx_step += 1 that _run_observed/_run_injection_only normally do first
        assert observer.begin_scope_calls == [('scope', 1, 0, False)]

    def test_idx_step_passed_only_for_agent_nodes(self):
        observer = FakeObserver()
        manager = make_manager(observer=observer)
        manager.idx_step = 5
        adapter = FakeAdapter(is_agent=False)

        manager._observe(lambda: 'ok', adapter, fault_applied=False)

        assert observer.begin_scope_calls[0][2] is None  # idx_step

    def test_can_restart_true_when_restart_configured_and_eager_allowed(self):
        observer = FakeObserver()
        manager = make_manager(observer=observer, restart=FakeRestart())
        adapter = FakeAdapter()

        manager._observe(lambda: 'ok', adapter, fault_applied=False, allow_eager=True)

        assert observer.begin_scope_calls[0][3] is True  # can_restart

    def test_can_restart_false_when_eager_not_allowed(self):
        observer = FakeObserver()
        manager = make_manager(observer=observer, restart=FakeRestart())
        adapter = FakeAdapter()

        manager._observe(lambda: 'ok', adapter, fault_applied=False, allow_eager=False)

        assert observer.begin_scope_calls[0][3] is False

    def test_records_injected_fault_when_configured(self):
        fault = FakeFault(idx_step=1)
        observer = FakeObserver()
        observer.record_injected_faults = True
        manager = make_manager(observer=observer, fault=fault)
        manager.idx_step = 1
        adapter = FakeAdapter(id='n1')

        manager._observe(lambda: 'ok', adapter, fault_applied=True)

        assert observer.record_injected_fault_calls == [(1, 'n1', fault)]

    def test_does_not_record_injected_fault_when_fault_not_applied(self):
        observer = FakeObserver()
        observer.record_injected_faults = True
        manager = make_manager(observer=observer)
        adapter = FakeAdapter()

        manager._observe(lambda: 'ok', adapter, fault_applied=False)

        assert observer.record_injected_fault_calls == []


class TestReportFaults:
    def test_records_a_detection_per_fault(self):
        manager = make_manager()
        adapter = FakeAdapter(id='n1', is_agent=True)
        manager.idx_step = 2
        faults = [make_detection(500, 'node'), make_detection(504, 'tool')]

        manager._report_faults(faults, adapter)

        recorded = take_detections()
        assert len(recorded) == 2
        assert recorded[0]['node_id'] == 'n1'
        assert recorded[0]['code'] == 500
        assert recorded[0]['idx_step'] == 2
        assert recorded[1]['code'] == 504

    def test_idx_step_none_for_non_agent_nodes(self):
        manager = make_manager()
        adapter = FakeAdapter(id='n1', is_agent=False)
        manager.idx_step = 2

        manager._report_faults([make_detection(500)], adapter)

        recorded = take_detections()
        assert recorded[0]['idx_step'] is None


class TestRestart:
    def test_without_observer_restores_and_reruns_once(self):
        restart = FakeRestart(restart_count=3, contexts=['ctx1'])
        checkpoint = FakeCheckpoint()
        manager = make_manager(restart=restart, checkpoint=checkpoint)
        adapter = FakeAdapter(id='n1')
        calls = []

        def callable_():
            calls.append(1)
            return 'ok'

        result = manager._restart(callable_, adapter)

        assert result == 'ok'
        assert len(calls) == 1
        assert checkpoint.restored == ['n1']
        assert adapter.set_input_calls == ['ctx1']

    def test_with_observer_stops_as_soon_as_clean(self):
        detections = [make_detection(500)]
        observer = FakeObserver(detections_by_call=[detections, []], restart_decision=True)
        restart = FakeRestart(restart_count=3, contexts=['ctx1', 'ctx2'])
        manager = make_manager(observer=observer, restart=restart)
        adapter = FakeAdapter(id='n1')

        result = manager._restart(lambda: 'ok', adapter)

        assert result == 'ok'
        assert adapter.set_input_calls == ['ctx1', 'ctx2']  # ran twice: still-faulty, then clean

    def test_exhausts_restart_budget_and_returns_last_result(self):
        detections = [make_detection(500)]
        observer = FakeObserver(detections_by_call=[detections], restart_decision=True)
        restart = FakeRestart(restart_count=2, contexts=['ctx1', 'ctx2'])
        manager = make_manager(observer=observer, restart=restart)
        adapter = FakeAdapter(id='n1')

        result = manager._restart(lambda: 'still-faulty', adapter)

        assert result == 'still-faulty'
        assert len(adapter.set_input_calls) == 2
        assert manager.logger.messages[-1][0].startswith('Restart budget')

    def test_last_attempt_disallows_eager_restart(self):
        observer = FakeObserver(detections_by_call=[[]])
        restart = FakeRestart(restart_count=1, contexts=['ctx1'])
        manager = make_manager(observer=observer, restart=restart)
        adapter = FakeAdapter(id='n1')

        manager._restart(lambda: 'ok', adapter)

        # attempt == max_restarts (1 == 1) -> allow_eager False -> can_restart False
        assert observer.begin_scope_calls[0][3] is False


class MessageAdapter(FakeAdapter):
    """FakeAdapter that also speaks the context <-> text protocol.

    Messages are `(role, text)` pairs, which is all a refined restart needs to
    show that roles survive compression.
    """

    def context_as_list(self, context=None):
        messages = self.input if context is None else context
        return [text for _, text in messages]

    def context_from_list(self, texts, context=None):
        messages = self.input if context is None else context
        return [
            (role, text or original)
            for (role, original), text in zip(messages, texts)
        ]


class HalvingCompressor(ContextCompressor):
    """Deterministic stand-in for LLMLingua: keeps the first half of the words."""

    def compress(self, contexts, question=''):
        self.question = question
        texts = [' '.join(c.split()[: max(1, len(c.split()) // 2)]) for c in contexts]
        return CompressionResult(texts, origin_tokens=10, compressed_tokens=5)


class TestRestartWithRefinedContext:
    """The refined-context strategy driven through the real manager flow."""

    def make(self, messages, restart_count=1):
        compressor = HalvingCompressor()
        restart = RestartRefinedContext(
            restart_count=restart_count, compressor=compressor
        )
        manager = make_manager(restart=restart)
        adapter = MessageAdapter(id='Coder', input=messages)
        return manager, restart, adapter, compressor

    def test_the_node_restarts_on_compressed_history_and_a_verbatim_last_message(self):
        messages = [
            ('user', 'the plan says do this and that'),
            ('assistant', 'the report says it is done'),
            ('user', 'the review says fix one more thing'),
        ]
        manager, _, adapter, _ = self.make(messages)

        manager.snapshot(adapter)
        manager._restart(lambda: 'ok', adapter)

        assert adapter.set_input_calls == [
            [
                ('user', 'the plan says'),
                ('assistant', 'the report says'),
                ('user', 'the review says fix one more thing'),
            ]
        ]

    def test_compression_is_conditioned_on_the_message_kept_verbatim(self):
        messages = [('user', 'the plan'), ('user', 'the review')]
        manager, _, adapter, compressor = self.make(messages)

        manager.snapshot(adapter)
        manager._restart(lambda: 'ok', adapter)

        assert compressor.question == 'the review'

    def test_the_snapshot_survives_the_fault_mangling_the_node_input(self):
        messages = [('user', 'the plan says do this'), ('user', 'the review')]
        manager, _, adapter, _ = self.make(messages)

        manager.snapshot(adapter)
        # What an injected fault does to the node before it runs.
        adapter.set_input([('user', 'CORRUPTED'), ('user', 'CORRUPTED')])
        adapter.set_input_calls.clear()

        manager._restart(lambda: 'ok', adapter)

        assert adapter.set_input_calls == [
            [('user', 'the plan'), ('user', 'the review')]
        ]


class FakeEdgeLink:
    def __init__(self, target):
        self.target = target


class EdgeRecorder:
    """Stands in for the framework method that hands a message to an edge.

    The framework calls it for *every* outgoing edge and evaluates the edge
    condition inside, so a node's captured target set is its whole static edge
    set -- it does not depend on which edge would actually fire.
    """

    def __init__(self):
        self.delivered = []

    def send(self, edge_link, msg, from_node, *args, **kwargs):
        self.delivered.append((edge_link.target, msg))


class TestRestartEdgeIsolation:
    """Only the attempt that ran last may reach the successors."""

    def _setup(self, attempts, restart_count=1):
        """`attempts` is one (edge targets, message) pair per node execution."""
        recorder = EdgeRecorder()
        suppressor = MethodSwapEdgeSuppressor(lambda *a, **kw: recorder, 'send')
        manager = make_manager(
            fault=FakeFault(idx_step=1),
            restart=FakeRestart(restart_count=restart_count, contexts=['restored']),
            edge_suppressor=suppressor,
        )
        remaining = list(attempts)

        def run_node():
            targets, msg = remaining.pop(0)
            for target in targets:
                recorder.send(FakeEdgeLink(target), msg, 'Planner')
            return msg

        return manager, recorder, run_node

    def test_the_restarted_output_replaces_the_faulty_one(self):
        # The common case, and the one the capture dict already handled: both
        # attempts touch the same edges, so the second write wins.
        manager, recorder, run_node = self._setup([
            (['FINAL', 'Coder'], 'faulty plan'),
            (['FINAL', 'Coder'], 'good plan'),
        ])

        manager._run_injection_only(run_node, FakeAdapter(id='Planner'))

        assert {msg for _, msg in recorder.delivered} == {'good plan'}

    def test_a_restart_that_emits_nothing_releases_nothing(self):
        # The asymmetry that does bite: a node produces no output at all, so
        # the restarted attempt touches no edge and overwrites nothing. Its
        # predecessor's captures would otherwise still be sitting there, and
        # the faulty message would be delivered as if it were the result.
        manager, recorder, run_node = self._setup([
            (['FINAL', 'Coder'], 'faulty plan'),
            ([], ''),
        ])

        manager._run_injection_only(run_node, FakeAdapter(id='Planner'))

        assert recorder.delivered == []

    def test_only_the_final_attempt_of_several_is_released(self):
        # Needs the Observer path: without one, _restart returns after a
        # single re-execution however large the restart budget.
        recorder = EdgeRecorder()
        attempts = [
            (['Coder'], 'faulty plan'),
            (['Coder'], 'second attempt'),
            ([], ''),
        ]
        manager = make_manager(
            restart=FakeRestart(restart_count=2, contexts=['c1', 'c2']),
            edge_suppressor=MethodSwapEdgeSuppressor(lambda *a, **kw: recorder, 'send'),
            observer=FakeObserver(
                detections_by_call=[[make_detection(500)], [make_detection(500)], []],
                restart_decision=True,
            ),
        )

        def run_node():
            targets, msg = attempts.pop(0)
            for target in targets:
                recorder.send(FakeEdgeLink(target), msg, 'Planner')
            return msg

        manager._run_observed(run_node, FakeAdapter(id='Planner'))

        assert attempts == []  # all three executions ran
        assert recorder.delivered == []

    def test_each_attempt_starts_from_a_clean_capture(self):
        edge_suppressor = FakeEdgeSuppressor()
        manager = make_manager(
            restart=FakeRestart(restart_count=3, contexts=['c1', 'c2', 'c3']),
            edge_suppressor=edge_suppressor,
            observer=FakeObserver(
                detections_by_call=[[make_detection(500)], [make_detection(500)], []],
                restart_decision=True,
            ),
        )

        manager._restart(lambda: 'ok', FakeAdapter(id='n1'), token='token')

        assert [call[0] for call in edge_suppressor.reset_calls] == ['token'] * 3

    def test_no_reset_is_attempted_when_nothing_was_suppressed(self):
        edge_suppressor = FakeEdgeSuppressor()
        manager = make_manager(restart=FakeRestart(), edge_suppressor=edge_suppressor)

        manager._restart(lambda: 'ok', FakeAdapter(id='n1'), token=None)

        assert edge_suppressor.reset_calls == []
