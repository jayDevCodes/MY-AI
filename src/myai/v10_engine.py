from __future__ import annotations

from .agent_runtime import AgentRuntimeResult, MultiModelAgentRuntime
from .schemas import ChatMessage
from .model_router import RoutingRequest
from .v9_engine import V9AIEngine


class V10AIEngine(V9AIEngine):
    """V10 self-healing runtime layered on the V9.1 cognitive mesh."""

    version = "v10.0"

    def repair_compute_policy(self, traceback_text: str):
        """Choose bounded compute from the real failure evidence before launching repair agents."""
        diagnosis = self.diagnose_failure(traceback_text)
        complexity = 0.75 if diagnosis.primary_frame else 0.6
        uncertainty = max(0.0, min(1.0, 1.0 - diagnosis.confidence))
        decision_request = RoutingRequest(
            task_kind="debugging",
            complexity=complexity,
            uncertainty=uncertainty,
            context_size=len(traceback_text),
            risk="high",
            latency_sensitive=False,
            quality_priority=True,
        )
        return self.compute_policy(decision_request)

    def repair_context_v10(self, traceback_text: str) -> tuple[ChatMessage, ...]:
        """Build compact repair context from V9 diagnosis plus V10 health/signature state."""
        diagnosis = self.diagnose_failure(traceback_text)
        base = list(super().repair_context_v9(traceback_text))
        signature = self.failure_signature(traceback_text)
        symbol = diagnosis.primary_frame.symbol if diagnosis.primary_frame else ""
        inspection = self.inspection_mode(symbol) if symbol else "deep"
        policy = self.repair_compute_policy(traceback_text)
        history = self.repair_memory.similar(
            diagnosis.error_type,
            diagnosis.message,
            limit=3,
        )
        base.append(
            ChatMessage(
                role="system",
                content=(
                    "V10 repair constraints:\n"
                    f"FAILURE SIGNATURE: {signature.value}\n"
                    f"CODE HEALTH INSPECTION MODE: {inspection}\n"
                    f"COMPUTE POLICY: tier={policy.preferred_tier}; depth={policy.reasoning_depth}; "
                    f"parallel={policy.max_parallel}; retries={policy.max_retries}; "
                    f"verification_passes={policy.verification_passes}\n"
                    f"SIMILAR FAILURES FOUND: {len(history)}\n"
                    "Preserve verified/stable code. Read only the causal impact slice. "
                    "A repair proposal is not a verified patch; execution and promotion remain validation-gated."
                ),
            )
        )
        return tuple(base)

    def propose_repair(self, traceback_text: str) -> AgentRuntimeResult:
        """Use the existing recursive multi-model runtime with V10's bounded compute policy."""
        if not self.settings.self_healing_enabled:
            return super().propose_repair(traceback_text)

        diagnosis = self.diagnose_failure(traceback_text)
        policy = self.repair_compute_policy(traceback_text)
        context = self.repair_context_v10(traceback_text)
        runtime = MultiModelAgentRuntime(
            self.model_pool,
            self.router,
            policy.execution_budget(base=self.agent_runtime.budget),
        )
        result = runtime.run(
            objective=(
                f"Repair {diagnosis.error_type} in "
                f"{diagnosis.primary_frame.path if diagnosis.primary_frame else 'unknown'} "
                f"with the smallest causally justified change. "
                f"Hypothesis: {diagnosis.root_cause_hypothesis}"
            ),
            task_kind="coding",
            context=context,
            state=self.cognitive_state,
        )
        self.cognitive_state.active_strategy = "v10-self-healing:" + policy.preferred_tier
        return result
