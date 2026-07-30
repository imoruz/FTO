from fto.adapters.probes.probe import ObservationProbe


class TestObservationProbe:
    def test_install_is_a_noop_by_default(self):
        probe = ObservationProbe()
        assert probe.install() is None
        assert probe.install(timeout_llm=1.0, timeout_tool=2.0) is None

    def test_uninstall_is_a_noop_by_default(self):
        probe = ObservationProbe()
        assert probe.uninstall() is None
