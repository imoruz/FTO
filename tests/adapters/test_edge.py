from fto.adapters.edge import EdgeSuppressor, MethodSwapEdgeSuppressor


class FakeEdgeLink:
    def __init__(self, target):
        self.target = target


class Recorder:
    def __init__(self):
        self.calls = []

    def send(self, edge_link, msg, from_node, *args, **kwargs):
        self.calls.append((edge_link, msg, from_node, args, kwargs))


class TestEdgeSuppressorBase:
    def test_suppress_is_a_noop_by_default(self):
        suppressor = EdgeSuppressor()
        assert suppressor.suppress() is None

    def test_restore_is_a_noop_by_default(self):
        suppressor = EdgeSuppressor()
        assert suppressor.restore(('token',)) is None


class TestMethodSwapEdgeSuppressor:
    def _suppressor(self, recorder):
        return MethodSwapEdgeSuppressor(lambda *a, **kw: recorder, 'send')

    def test_suppress_replaces_the_method_and_captures_calls(self):
        recorder = Recorder()
        suppressor = self._suppressor(recorder)

        suppressor.suppress()
        recorder.send(FakeEdgeLink('target-a'), 'hello', 'node-a')

        # The real method never ran while suppressed.
        assert recorder.calls == []

    def test_restore_replays_captured_calls_in_first_seen_order(self):
        recorder = Recorder()
        suppressor = self._suppressor(recorder)
        link_a = FakeEdgeLink('target-a')
        link_b = FakeEdgeLink('target-b')

        token = suppressor.suppress()
        recorder.send(link_a, 'first', 'node-a')
        recorder.send(link_b, 'second', 'node-b')
        suppressor.restore(token)

        assert [call[1] for call in recorder.calls] == ['first', 'second']

    def test_restore_keeps_only_the_last_message_per_target(self):
        recorder = Recorder()
        suppressor = self._suppressor(recorder)
        link_a1 = FakeEdgeLink('target-a')
        link_a2 = FakeEdgeLink('target-a')  # distinct edge_link, same target object identity matters via id()
        link_a2.target = link_a1.target
        link_b = FakeEdgeLink('target-b')

        token = suppressor.suppress()
        recorder.send(link_a1, 'stale', 'node-a')
        recorder.send(link_b, 'kept-b', 'node-b')
        recorder.send(link_a2, 'fresh', 'node-a')  # same target as link_a1 -> overwrites, but order preserved
        suppressor.restore(token)

        assert len(recorder.calls) == 2
        assert recorder.calls[0][1] == 'fresh'
        assert recorder.calls[1][1] == 'kept-b'

    def test_restore_restores_the_original_method(self):
        recorder = Recorder()
        suppressor = self._suppressor(recorder)

        token = suppressor.suppress()
        suppressor.restore(token)
        recorder.send(FakeEdgeLink('target-a'), 'direct', 'node-a')

        assert len(recorder.calls) == 1
        assert recorder.calls[0][1] == 'direct'

    def test_restore_with_no_captured_calls_leaves_recorder_untouched(self):
        recorder = Recorder()
        suppressor = self._suppressor(recorder)

        token = suppressor.suppress()
        suppressor.restore(token)

        assert recorder.calls == []


class TestEdgeSuppressorBaseReset:
    def test_reset_is_a_noop_by_default(self):
        assert EdgeSuppressor().reset(('token',)) is None


class TestMethodSwapEdgeSuppressorReset:
    def _suppressor(self, recorder):
        return MethodSwapEdgeSuppressor(lambda *a, **kw: recorder, 'send')

    def test_reset_drops_what_was_captured_so_far(self):
        recorder = Recorder()
        suppressor = self._suppressor(recorder)

        token = suppressor.suppress()
        recorder.send(FakeEdgeLink('target-a'), 'from the faulty attempt', 'node-a')
        suppressor.reset(token)
        suppressor.restore(token)

        assert recorder.calls == []

    def test_reset_keeps_suppression_on(self):
        recorder = Recorder()
        suppressor = self._suppressor(recorder)

        token = suppressor.suppress()
        suppressor.reset(token)
        recorder.send(FakeEdgeLink('target-a'), 'after reset', 'node-a')

        # Still withheld, not passed through to the real method.
        assert recorder.calls == []

    def test_only_calls_made_after_the_reset_are_replayed(self):
        recorder = Recorder()
        suppressor = self._suppressor(recorder)

        token = suppressor.suppress()
        recorder.send(FakeEdgeLink('exit-node'), 'faulty: we are done', 'planner')
        suppressor.reset(token)
        recorder.send(FakeEdgeLink('coder'), 'restarted: here is the plan', 'planner')
        suppressor.restore(token)

        # Without the reset the faulty "we are done" would reach the exit node
        # alongside the restarted node's real output.
        assert [call[1] for call in recorder.calls] == ['restarted: here is the plan']

    def test_reset_clears_the_replay_order_too(self):
        recorder = Recorder()
        suppressor = self._suppressor(recorder)
        link_a, link_b = FakeEdgeLink('target-a'), FakeEdgeLink('target-b')

        token = suppressor.suppress()
        recorder.send(link_a, 'stale-a', 'node')
        recorder.send(link_b, 'stale-b', 'node')
        suppressor.reset(token)
        recorder.send(link_b, 'fresh-b', 'node')
        recorder.send(link_a, 'fresh-a', 'node')
        suppressor.restore(token)

        assert [call[1] for call in recorder.calls] == ['fresh-b', 'fresh-a']
