from __future__ import annotations

from .agent_runtime import AgentRuntimeResult
from .schemas import ChatMessage
from .v9_engine import V9AIEngine
from .model_router import RoutingRequest


class V10AIEngine(V9AIEngine):
    """V10 self-healing runtime layered on the V9.1 cognitive mesh."""

    version = "v10.0"

    def repair_context_v10(self, traceback_text: str) -> tuple[ChatMessage, ...]:
        """Extend V9's targeted repair context with V10 health/signature/compute evidence."""
        diagnosis = self.diagnose_failure(traceback_text)
        context = list(self.repair_context_v9(traceback_text))
        symbol = diagnosis.primary_frame.symbol if diagnosis.primary_frame else ""
        signature = self.failure_signature(traceback_text)
        policy = self.compute_policy(
            RoutingRequest(
                task_kind="coding",
                complexity=min(1.0, 0.75 + (0.2 if diagnosis.impact.nodes else 0.0)),
                uncertainty=max(0.0, 1.0 - diagnosis.confidence),
                risk="high",
            )
        )
        prior = self.failure_signatures.similar(signature, limit=3)
        trace_count = 0
        if diagnosis.primary_frame:
            trace_count = sum(
                1
                for event in self.runtime_traces.events.values()
                if event.path == diagnosis.primary_frame.path
                and (event.line is None or abs(event.line - diagnosis.primary_frame.line) <= 8)
            )
        context.append(
            ChatMessage(
                role="system",
                content=(
                    "V10 repair control plane: preserve verified code, use causal evidence first, "
                    "and prefer the smallest justified repair. Do not promote code automatically."
                ),
            )
        )
        context.append(
            ChatMessage(
                role="user",
                content=(
                    f"FAILURE SIGNATURE: {signature.value}\n"
                    f"CODE HEALTH MODE: {self.inspection_mode(symbol) if symbol else 'deep'}\n"
                    f"TRACE EVENTS NEAR FAILURE: {trace_count}\n"
                    f"SIMILAR FAILURE SIGNATURES: {len(prior)}\n"
                    f"COMPUTE POLICY: depth={policy.reasoning_depth}; parallel={policy.max_parallel}; "
                    f"retries={policy.max_retries}; verification={policy.verification_passes}; tier={policy.preferred_tier}\n"
                    f"REPAIR ATTEMPTS BUDGET: {self.settings.self_healing_max_repair_attempts}"
                ),
            )
        )
        return tuple(context)

    def propose_repair(self, traceback_text: str) -> AgentRuntimeResult:
        """Run V10 repair through the existing multi-model runtime using bounded evidence context."""
        diagnosis = self.diagnose_failure(traceback_text)
        context = self.repair_context_v10(traceback_text)
        return self.agent_runtime.run(
            objective=(
                f"Repair {diagnosis.error_type} with the minimum causally justified change. "
                f"Primary symbol: {diagnosis.primary_frame.symbol if diagnosis.primary_frame else 'unknown'}. "
                f"Confidence: {diagnosis.confidence:.2f}."
            ),
            task_kind="coding",
            context=context,
        )
