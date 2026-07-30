from __future__ import annotations

import concurrent.futures
import contextvars

from fto.adapters.probes.probe import ObservationProbe
from fto.detection.observer import DetectionSink, classify_exception
from fto.detection.classifier import classify_tool_result
from fto.detection.detection import Detection
from fto.detection.atp import status


class ChatDevProbe(ObservationProbe):
    def __init__(self):
        self._orig = {}

    def install(self, timeout_llm: float | None = None, timeout_tool: float | None = None):
        from runtime.node.executor.agent_executor import AgentNodeExecutor
        from runtime.node.agent.tool.tool_manager import ToolManager

        inv = AgentNodeExecutor._invoke_provider
        tool = ToolManager.execute_tool

        def _invoke_provider(self, provider, client, conv, timeline, opts, specs, node):
            if timeout_llm is not None:
                ctx = contextvars.copy_context()
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                fut = executor.submit(ctx.run, inv, self, provider, client, conv, timeline, opts, specs, node)
                try:
                    return fut.result(timeout=timeout_llm)
                except concurrent.futures.TimeoutError:
                    executor.shutdown(wait=False)
                    DetectionSink.record(Detection(status(504), 'llm', detail=f'llm hang > {timeout_llm}s'))
                    raise TimeoutError(f'LLM call timed out after {timeout_llm}s')
                except BaseException as exc:
                    executor.shutdown(wait=False)
                    DetectionSink.record(classify_exception(exc, 'llm'))
                    raise
            try:
                return inv(self, provider, client, conv, timeline, opts, specs, node)
            except BaseException as exc:
                DetectionSink.record(classify_exception(exc, 'llm'))
                raise

        async def execute_tool(self, tool_name, arguments, tool_config, *, tool_context=None):
            import asyncio
            if timeout_tool is not None:
                ctx = contextvars.copy_context()

                def _run_in_new_loop():
                    # Tools often do blocking sync I/O with no await points, so asyncio.wait_for
                    # on the raw coroutine can't interrupt them (the loop is blocked). Running in
                    # a thread keeps the main loop free so the timeout callback can actually fire.
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            tool(self, tool_name, arguments, tool_config, tool_context=tool_context)
                        )
                    finally:
                        new_loop.close()
                        asyncio.set_event_loop(None)

                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                current_loop = asyncio.get_event_loop()
                fut = current_loop.run_in_executor(executor, ctx.run, _run_in_new_loop)
                try:
                    result = await asyncio.wait_for(fut, timeout=timeout_tool)
                except asyncio.TimeoutError:
                    executor.shutdown(wait=False)
                    DetectionSink.record(Detection(status(504), 'tool', detail=f'tool hang > {timeout_tool}s'))
                    raise TimeoutError(f'Tool call timed out after {timeout_tool}s')
                except BaseException as exc:
                    executor.shutdown(wait=False)
                    DetectionSink.record(classify_exception(exc, 'tool'))
                    raise
            else:
                try:
                    result = await tool(self, tool_name, arguments, tool_config, tool_context=tool_context)
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
        