from fto.adapters.probes.probe import ObservationProbe
from fto.detection.observer import DetectionSink, classify_exception
from fto.detection.classifier import classify_tool_result


class ChatDevProbe(ObservationProbe):
    def __init__(self):
        self._orig = {}

    def install(self):
        from runtime.node.executor.agent_executor import AgentNodeExecutor
        from runtime.node.agent.tool.tool_manager import ToolManager

        inv = AgentNodeExecutor._invoke_provider
        tool = ToolManager.execute_tool

        def _invoke_provider(self, provider, client, conv, timeline, opts, specs, node):
            try:
                return inv(self, provider, client, conv, timeline, opts, specs, node)
            except BaseException as exc:                       # noqa: BLE001
                DetectionSink.record(classify_exception(exc, 'llm'))
                raise

        async def execute_tool(self, tool_name, arguments, tool_config, *, tool_context=None):
            try:
                result = await tool(self, tool_name, arguments, tool_config,
                                    tool_context=tool_context)
            except BaseException as exc:
                DetectionSink.record(classify_exception(exc, 'tool'))
                raise
            hit = classify_tool_result(tool_name, result)
            if hit:
                DetectionSink.record(hit)
            return result

        self._orig = {'inv': (AgentNodeExecutor, inv), 'tool': (ToolManager, tool)}
        AgentNodeExecutor._invoke_provider = _invoke_provider
        ToolManager.execute_tool = execute_tool

    def uninstall(self):
        if not self._orig:
            return
        cls_i, inv = self._orig['inv'] 
        cls_i._invoke_provider = inv
        cls_t, tool = self._orig['tool'] 
        cls_t.execute_tool = tool
        self._orig = {}