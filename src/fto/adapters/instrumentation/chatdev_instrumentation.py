from collections import defaultdict
import uuid
from opentelemetry import context as otel_context
from llmmas_otel.span_factory import default_span_factory as sf

from entity.configs.node.node import Node
from runtime.node.agent.providers.base import ModelProvider

_PATCHED = False


def apply_patches():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from workflow.graph import GraphExecutor
    from runtime.node.executor.agent_executor import AgentNodeExecutor
    from runtime.node.agent.tool.tool_manager import ToolManager
    from workflow.executor.parallel_executor import ParallelExecutor

    _exe = GraphExecutor._execute
    _exe_node = GraphExecutor._execute_node
    _proc = GraphExecutor._process_edge_output
    _invoke = AgentNodeExecutor._invoke_provider
    _tool = ToolManager.execute_tool
    _parallel = ParallelExecutor._execute_parallel_batch

    def _execute(self, task):
        self._otel_steps = defaultdict(int)
        with sf.session(session_id=f"{self.graph.name}::{uuid.uuid4().hex[:8]}") as session_span:
            # probe = SystemMetricsProbe()
            # probe.start(session_span)
            # try:
            _exe(self, task)
            # finally:
            #     probe.stop()

    def _execute_node(self, node: Node):
        idx = self._otel_steps[node.id]
        self._otel_steps[node.id] = idx + 1

        def _run_with_receives(ctx_manager):
            with ctx_manager:
                # Emit each receive span as an instant marker (immediately closed)
                # so they appear as siblings rather than nesting the execution.
                for msg in node.input:
                    src = msg.metadata.get("source")
                    if src and src not in (node.id, "TASK"):
                        carrier = {k: v for k, v in msg.metadata.items() if k == "_otel_traceparent_"}
                        with sf.a2a_receive(
                            source_agent_id=src,
                            target_agent_id=node.id,
                            edge_id=f"{src}->{node.id}",
                            message_id=msg.metadata.get("_otel_msg_id", f"msg-{uuid.uuid4().hex[:12]}"),
                            carrier=carrier or None,
                            link_from_carrier=bool(carrier),
                        ):
                            pass
                _exe_node(self, node)

        if node.node_type == "agent":
            _run_with_receives(sf.agent_step(agent_id=node.id, step_index=idx))
        else:
            _exe_node(self, node)

    def _process_edge_output(self, edge_link, msg, from_node):
        tgt = edge_link.target
        mid = msg.metadata.get("_otel_msg_id") or f"msg-{uuid.uuid4().hex[:12]}"
        carrier = {}
        with sf.a2a_send(
            source_agent_id=from_node.id,
            target_agent_id=tgt.id,
            edge_id=f"{from_node.id}->{tgt.id}",
            message_id=mid,
            carrier=carrier,
            propagate_context=True,
        ):
            if carrier:
                msg.metadata.update({k: v for k, v in carrier.items()})
                msg.metadata["_otel_msg_id"] = mid
            _proc(self, edge_link, msg, from_node)

    def _invoke_provider(self, provider, client, conv, timeline, opts, specs, node):
        with sf.llm_call(
            provider_name=provider.provider,
            model=provider.model_name
        ) as ctx:
            from llmmas_otel.injection import DecisionKind
            from opentelemetry.trace.status import Status, StatusCode
            dec = ctx.decision
            if dec is not None and dec.kind == DecisionKind.RAISE:
                exc = dec.raise_exception or RuntimeError("Injected LLM error")
                ctx.span.record_exception(exc)
                ctx.span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise exc
            if dec is not None and dec.kind == DecisionKind.RETURN:
                return dec.return_value
            response = _invoke(self, provider, client, conv, timeline, opts, specs, node)
            # Attach token counts to the span so Jaeger shows them per LLM call.
            if response is not None and response.raw_response is not None:
                try:
                    # NOTE: ONLY COMPATIBLE WITH GRAZIE FOR NOW!!
                    usage = provider.get_quota(response.raw_response)
                    if usage:
                        ctx.span.set_attribute("gen_ai.usage.total_tokens", usage.amount)
                except Exception:
                    pass
            return response

    async def execute_tool(self, tool_name, arguments, tool_config, *, tool_context=None):
        with sf.tool_call(tool_name=tool_name, tool_type=getattr(tool_config, "type", None)):
            return await _tool(self, tool_name, arguments, tool_config, tool_context=tool_context)

    def _execute_parallel_batch(self, items, executor_func, item_desc_func):
        ctx = otel_context.get_current()
        def _with_ctx(item):
            token = otel_context.attach(ctx)
            try:
                return executor_func(item)
            finally:
                otel_context.detach(token)
        _parallel(self, items, _with_ctx, item_desc_func)

    GraphExecutor._execute = _execute
    GraphExecutor._execute_node = _execute_node
    GraphExecutor._process_edge_output = _process_edge_output
    AgentNodeExecutor._invoke_provider = _invoke_provider
    ToolManager.execute_tool = execute_tool
    ParallelExecutor._execute_parallel_batch = _execute_parallel_batch
