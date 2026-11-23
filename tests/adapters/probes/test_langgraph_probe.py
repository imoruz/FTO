import time

import pytest

from fto.adapters.probes.langgraph import LangGraphProbe


class FakeChatGrazie:
    """Stand-in for experiments.langgraph.providers.ChatGrazie."""

    def __init__(self, behavior=None):
        self.behavior = behavior or (lambda messages: f'generated:{messages}')

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self.behavior(messages)


class FakeBaseTool:
    """Stand-in for langchain_core.tools.BaseTool."""

    def __init__(self, name='my-tool', behavior=None):
        self.name = name
        self.behavior = behavior or (lambda input: f'result:{input}')

    def invoke(self, input, config=None, **kwargs):
        return self.behavior(input)


@pytest.fixture
def stubbed_langgraph_deps(fake_module):
    fake_module('experiments.langgraph.providers', ChatGrazie=FakeChatGrazie)
    fake_module('langchain_core.tools', BaseTool=FakeBaseTool)
    return FakeChatGrazie, FakeBaseTool


class TestLangGraphProbeInstallUninstall:
    def test_install_patches_generate_and_invoke(self, stubbed_langgraph_deps):
        probe = LangGraphProbe()
        original_generate = FakeChatGrazie._generate
        original_invoke = FakeBaseTool.invoke

        probe.install()

        assert FakeChatGrazie._generate is not original_generate
        assert FakeBaseTool.invoke is not original_invoke

        probe.uninstall()

    def test_uninstall_restores_original_methods(self, stubbed_langgraph_deps):
        probe = LangGraphProbe()
        original_generate = FakeChatGrazie._generate
        original_invoke = FakeBaseTool.invoke

        probe.install()
        probe.uninstall()

        assert FakeChatGrazie._generate is original_generate
        assert FakeBaseTool.invoke is original_invoke

    def test_uninstall_before_install_is_a_noop(self, stubbed_langgraph_deps):
        probe = LangGraphProbe()
        probe.uninstall()  # should not raise


class TestLangGraphProbeGenerate:
    def test_successful_generate_returns_original_result(self, stubbed_langgraph_deps, detection_scope):
        probe = LangGraphProbe()
        probe.install()
        try:
            llm = FakeChatGrazie()
            result = llm._generate(['hello'])
            assert result == "generated:['hello']"
            assert detection_scope() == []
        finally:
            probe.uninstall()

    def test_generate_exception_is_recorded_and_reraised(self, stubbed_langgraph_deps, detection_scope):
        probe = LangGraphProbe()
        probe.install()
        try:
            llm = FakeChatGrazie(behavior=lambda messages: (_ for _ in ()).throw(ValueError('boom')))
            with pytest.raises(ValueError, match='boom'):
                llm._generate(['hello'])
            detections = detection_scope()
            assert len(detections) == 1
            assert detections[0].site == 'llm'
        finally:
            probe.uninstall()

    def test_generate_timeout_raises_and_records_504(self, stubbed_langgraph_deps, detection_scope):
        probe = LangGraphProbe()
        probe.install(timeout_llm=0.05)
        try:
            llm = FakeChatGrazie(behavior=lambda messages: time.sleep(0.5))
            with pytest.raises(TimeoutError):
                llm._generate(['hello'])
            detections = detection_scope()
            assert len(detections) == 1
            assert detections[0].atp.code == 504
            assert detections[0].site == 'llm'
        finally:
            probe.uninstall()


class TestLangGraphProbeInvoke:
    def test_successful_invoke_returns_original_result(self, stubbed_langgraph_deps, detection_scope):
        probe = LangGraphProbe()
        probe.install()
        try:
            tool = FakeBaseTool(name='search')
            result = tool.invoke({'q': 'x'})
            assert result == "result:{'q': 'x'}"
            assert detection_scope() == []
        finally:
            probe.uninstall()

    def test_invoke_exception_is_recorded_and_reraised(self, stubbed_langgraph_deps, detection_scope):
        probe = LangGraphProbe()
        probe.install()
        try:
            tool = FakeBaseTool(name='search', behavior=lambda i: (_ for _ in ()).throw(RuntimeError('kaboom')))
            with pytest.raises(RuntimeError, match='kaboom'):
                tool.invoke({'q': 'x'})
            detections = detection_scope()
            assert len(detections) == 1
            assert detections[0].site == 'tool'
        finally:
            probe.uninstall()

    def test_invoke_timeout_raises_and_records_504(self, stubbed_langgraph_deps, detection_scope):
        probe = LangGraphProbe()
        probe.install(timeout_tool=0.05)
        try:
            tool = FakeBaseTool(name='search', behavior=lambda i: time.sleep(0.5))
            with pytest.raises(TimeoutError):
                tool.invoke({'q': 'x'})
            detections = detection_scope()
            assert len(detections) == 1
            assert detections[0].atp.code == 504
            assert detections[0].site == 'tool'
        finally:
            probe.uninstall()

    def test_invoke_classifies_error_shaped_tool_result(self, stubbed_langgraph_deps, detection_scope):
        probe = LangGraphProbe()
        probe.install()
        try:
            tool = FakeBaseTool(name='search', behavior=lambda i: "ERROR calling 'search': No module named 'x'")
            result = tool.invoke({'q': 'x'})
            assert result == "ERROR calling 'search': No module named 'x'"
            detections = detection_scope()
            assert len(detections) == 1
            assert detections[0].atp.code == 522
        finally:
            probe.uninstall()
