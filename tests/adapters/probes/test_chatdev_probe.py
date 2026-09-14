import asyncio

import pytest

from fto.adapters.probes.chatdev import ChatDevProbe
from tests.conftest import detection_scope_cm


class FakeAgentNodeExecutor:
    """Stand-in for runtime.node.executor.agent_executor.AgentNodeExecutor."""

    def __init__(self, behavior=None):
        self.behavior = behavior or (lambda provider: f'invoked:{provider}')

    def _invoke_provider(self, provider, client, conv, timeline, opts, specs, node):
        return self.behavior(provider)


class FakeToolManager:
    """Stand-in for runtime.node.agent.tool.tool_manager.ToolManager."""

    def __init__(self, behavior=None):
        self.behavior = behavior or (lambda tool_name, arguments: f'result:{tool_name}:{arguments}')

    async def execute_tool(self, tool_name, arguments, tool_config, *, tool_context=None):
        result = self.behavior(tool_name, arguments)
        if asyncio.iscoroutine(result):
            return await result
        return result


@pytest.fixture
def stubbed_chatdev_deps(fake_module):
    fake_module('runtime.node.executor.agent_executor', AgentNodeExecutor=FakeAgentNodeExecutor)
    fake_module('runtime.node.agent.tool.tool_manager', ToolManager=FakeToolManager)
    return FakeAgentNodeExecutor, FakeToolManager


class TestChatDevProbeInstallUninstall:
    def test_install_patches_invoke_provider_and_execute_tool(self, stubbed_chatdev_deps):
        probe = ChatDevProbe()
        original_invoke = FakeAgentNodeExecutor._invoke_provider
        original_execute = FakeToolManager.execute_tool

        probe.install()

        assert FakeAgentNodeExecutor._invoke_provider is not original_invoke
        assert FakeToolManager.execute_tool is not original_execute

        probe.uninstall()

    def test_uninstall_restores_original_methods(self, stubbed_chatdev_deps):
        probe = ChatDevProbe()
        original_invoke = FakeAgentNodeExecutor._invoke_provider
        original_execute = FakeToolManager.execute_tool

        probe.install()
        probe.uninstall()

        assert FakeAgentNodeExecutor._invoke_provider is original_invoke
        assert FakeToolManager.execute_tool is original_execute

    def test_uninstall_before_install_is_a_noop(self, stubbed_chatdev_deps):
        probe = ChatDevProbe()
        probe.uninstall()  # should not raise


class TestChatDevProbeInvokeProvider:
    def test_successful_invoke_returns_original_result(self, stubbed_chatdev_deps, detection_scope):
        probe = ChatDevProbe()
        probe.install()
        try:
            executor = FakeAgentNodeExecutor()
            result = executor._invoke_provider('p', 'c', 'conv', 'tl', 'opts', 'specs', 'node')
            assert result == 'invoked:p'
            assert detection_scope() == []
        finally:
            probe.uninstall()

    def test_invoke_exception_is_recorded_and_reraised(self, stubbed_chatdev_deps, detection_scope):
        probe = ChatDevProbe()
        probe.install()
        try:
            executor = FakeAgentNodeExecutor(behavior=lambda p: (_ for _ in ()).throw(ValueError('boom')))
            with pytest.raises(ValueError, match='boom'):
                executor._invoke_provider('p', 'c', 'conv', 'tl', 'opts', 'specs', 'node')
            detections = detection_scope()
            assert len(detections) == 1
            assert detections[0].site == 'llm'
        finally:
            probe.uninstall()

    def test_invoke_timeout_raises_and_records_504(self, stubbed_chatdev_deps, detection_scope):
        import time

        probe = ChatDevProbe()
        probe.install(timeout_llm=0.05)
        try:
            executor = FakeAgentNodeExecutor(behavior=lambda p: time.sleep(0.5))
            with pytest.raises(TimeoutError):
                executor._invoke_provider('p', 'c', 'conv', 'tl', 'opts', 'specs', 'node')
            detections = detection_scope()
            assert len(detections) == 1
            assert detections[0].atp.code == 504
            assert detections[0].site == 'llm'
        finally:
            probe.uninstall()


class TestChatDevProbeExecuteTool:
    async def test_successful_execute_returns_original_result(self, stubbed_chatdev_deps):
        probe = ChatDevProbe()
        probe.install()
        try:
            with detection_scope_cm() as get_detections:
                manager = FakeToolManager()
                result = await manager.execute_tool('search', {'q': 'x'}, {})
                assert result == "result:search:{'q': 'x'}"
                assert get_detections() == []
        finally:
            probe.uninstall()

    async def test_execute_exception_is_recorded_and_reraised(self, stubbed_chatdev_deps):
        probe = ChatDevProbe()
        probe.install()
        try:
            def boom(tool_name, arguments):
                raise RuntimeError('kaboom')

            manager = FakeToolManager(behavior=boom)
            with detection_scope_cm() as get_detections:
                with pytest.raises(RuntimeError, match='kaboom'):
                    await manager.execute_tool('search', {'q': 'x'}, {})
                detections = get_detections()
                assert len(detections) == 1
                assert detections[0].site == 'tool'
        finally:
            probe.uninstall()

    async def test_execute_timeout_raises_and_records_504(self, stubbed_chatdev_deps):
        def slow(tool_name, arguments):
            import time
            time.sleep(0.5)
            return 'too-late'

        probe = ChatDevProbe()
        probe.install(timeout_tool=0.05)
        try:
            manager = FakeToolManager(behavior=slow)
            with detection_scope_cm() as get_detections:
                with pytest.raises(TimeoutError):
                    await manager.execute_tool('search', {'q': 'x'}, {})
                detections = get_detections()
                assert len(detections) == 1
                assert detections[0].atp.code == 504
                assert detections[0].site == 'tool'
        finally:
            probe.uninstall()

    async def test_execute_classifies_error_shaped_tool_result(self, stubbed_chatdev_deps):
        probe = ChatDevProbe()
        probe.install()
        try:
            manager = FakeToolManager(behavior=lambda tool_name, arguments: "ERROR calling 'search': boom")
            with detection_scope_cm() as get_detections:
                result = await manager.execute_tool('search', {'q': 'x'}, {})
                assert result == "ERROR calling 'search': boom"
                detections = get_detections()
                assert len(detections) == 1
                assert detections[0].atp.code == 500
        finally:
            probe.uninstall()
