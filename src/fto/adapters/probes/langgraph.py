from fto.adapters.probes.probe import ObservationProbe
from fto.detection.observer import DetectionSink
from fto.detection.classifier import classify_exception, classify_tool_result


class LangGraphProbe(ObservationProbe):

    def __init__(self):
        self._orig = {}

    def install(self):
        from experiments.langgraph.providers import ChatGrazie
        from langchain_core.tools import BaseTool

        gen = ChatGrazie._generate
        inv = BaseTool.invoke

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            try:
                return gen(self, messages, stop=stop, run_manager=run_manager, **kwargs)
            except BaseException as exc:
                DetectionSink.record(classify_exception(exc, 'llm'))
                raise

        def invoke(self, input, config=None, **kwargs):
            tool_name = getattr(self, 'name', 'unknown')
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
