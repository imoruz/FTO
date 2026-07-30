from __future__ import annotations

import concurrent.futures
import contextvars

from fto.adapters.probes.probe import ObservationProbe
from fto.detection.observer import DetectionSink
from fto.detection.classifier import classify_exception, classify_tool_result
from fto.detection.detection import Detection
from fto.detection.atp import status


class LangGraphProbe(ObservationProbe):

    def __init__(self):
        self._orig = {}

    def install(self, timeout_llm: float | None = None, timeout_tool: float | None = None):
        from experiments.langgraph.providers import ChatGrazie
        from langchain_core.tools import BaseTool

        gen = ChatGrazie._generate
        inv = BaseTool.invoke

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if timeout_llm is not None:
                ctx = contextvars.copy_context()
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                fut = executor.submit(ctx.run, gen, self, messages, stop=stop, run_manager=run_manager, **kwargs)
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
                return gen(self, messages, stop=stop, run_manager=run_manager, **kwargs)
            except BaseException as exc:
                DetectionSink.record(classify_exception(exc, 'llm'))
                raise

        def invoke(self, input, config=None, **kwargs):
            tool_name = getattr(self, 'name', 'unknown')
            if timeout_tool is not None:
                ctx = contextvars.copy_context()
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                fut = executor.submit(ctx.run, inv, self, input, config=config, **kwargs)
                try:
                    result = fut.result(timeout=timeout_tool)
                except concurrent.futures.TimeoutError:
                    executor.shutdown(wait=False)
                    DetectionSink.record(Detection(status(504), 'tool', detail=f'tool hang > {timeout_tool}s'))
                    raise TimeoutError(f'Tool call timed out after {timeout_tool}s')
                except BaseException as exc:
                    executor.shutdown(wait=False)
                    DetectionSink.record(classify_exception(exc, 'tool'))
                    raise
            else:
                try:
                    result = inv(self, input, config=config, **kwargs)
                except BaseException as exc:
                    DetectionSink.record(classify_exception(exc, 'tool'))
                    raise
            hit = classify_tool_result(tool_name, result)
            if hit:
                DetectionSink.record(hit)
            return result

        self._orig = {'gen': (ChatGrazie, gen), 'inv': (BaseTool, inv)}
        ChatGrazie._generate = _generate
        BaseTool.invoke = invoke

    def uninstall(self):
        if not self._orig:
            return
        cls_g, gen = self._orig['gen']
        cls_g._generate = gen

        cls_i, inv = self._orig['inv']
        cls_i.invoke = inv
        self._orig = {}
