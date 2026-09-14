from fto.recovery.restart import Restart, RestartAllContext, RestartNoContext, RestartRefinedContext


class TestRestartBase:
    def test_default_restart_count_is_one(self):
        assert Restart().restart_count == 1

    def test_restart_count_is_configurable(self):
        assert Restart(restart_count=5).restart_count == 5

    def test_set_context_deep_copies_the_context(self):
        restart = Restart()
        original = {'a': [1, 2, 3]}

        restart.set_context(original)
        original['a'].append(4)

        assert restart.context == {'a': [1, 2, 3]}
        assert restart.context is not original

    def test_get_context_is_a_stub_returning_none(self):
        restart = Restart()
        restart.set_context({'a': 1})
        assert restart.get_context() is None


class TestRestartAllContext:
    def test_get_context_returns_the_stored_context(self):
        restart = RestartAllContext(restart_count=3)
        restart.set_context({'history': ['step1', 'step2']})

        assert restart.get_context() == {'history': ['step1', 'step2']}

    def test_get_context_returns_the_deep_copied_object(self):
        restart = RestartAllContext()
        original = {'a': [1]}
        restart.set_context(original)

        assert restart.get_context() is restart.context
        assert restart.get_context() is not original


class TestRestartNoContext:
    def test_get_context_always_returns_none(self):
        restart = RestartNoContext()
        restart.set_context({'history': ['step1', 'step2']})

        assert restart.get_context() is None

    def test_get_context_returns_none_even_without_set_context(self):
        restart = RestartNoContext()
        assert restart.get_context() is None


class TestRestartRefinedContext:
    def test_get_context_is_not_yet_implemented(self):
        restart = RestartRefinedContext()
        restart.set_context({'history': ['step1']})

        assert restart.get_context() is None

    def test_default_restart_count_is_one(self):
        assert RestartRefinedContext().restart_count == 1
