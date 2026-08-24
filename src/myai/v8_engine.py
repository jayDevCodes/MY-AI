from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from .agent_runtime import AgentRuntimeResult
from .engine import AIEngine
from .repository_twin import CausalRepositoryTwin
from .schemas import ChatMessage
from .self_healing import CausalDiagnosis, CausalErrorEngine, RepairMemory, RepairMemoryRecord


class V8AIEngine(AIEngine):
    """V8 engine with causal repository intelligence and repair experience memory."""

    version = "v8.0"

    def __init__(self) -> None:
        super().__init__()
        self.repository_twin = CausalRepositoryTwin(self.code_index)
        self.repair_memory = RepairMemory(self.settings.repair_memory_path)
        self.causal_engine = CausalErrorEngine(self.repository_twin, self.repair_memory)
        self._refresh_twin()

    def _refresh_twin(self) -> None:
        self._ensure_code_index_fresh()
        self.repository_twin.rebuild(self.settings.code_index_root)

    def refresh_repository_twin(self) -> None:
        self.refresh_code_index()
        self.repository_twin.rebuild(self.settings.code_index_root)

    def diagnose_failure(self, traceback_text: str) -> CausalDiagnosis:
        self._refresh_twin()
        return self.causal_engine.diagnose(traceback_text)

    def repair_context(self, traceback_text: str) -> tuple[ChatMessage, ...]:
        diagnosis = self.diagnose_failure(traceback_text)
        history = self.repair_memory.similar(diagnosis.error_type, diagnosis.message, limit=3)
        impact = diagnosis.impact
        compact = [
            ChatMessage(
                role="system",
                content=(
                    "You are MY-AI V8 repair specialist. Use deterministic runtime evidence first. "
                    "Do not request or reread unrelated repository files. Produce a minimal patch plan "
                    "or unified diff limited to the diagnosed impact slice. Preserve known-good code."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"ERROR: {diagnosis.error_type}: {diagnosis.message}\n"
                    f"ROOT-CAUSE HYPOTHESIS: {diagnosis.root_cause_hypothesis}\n"
                    f"CONFIDENCE: {diagnosis.confidence:.2f}\n"
                    f"AFFECTED FILES: {', '.join(diagnosis.affected_files)}\n"
                    f"EVIDENCE: {' | '.join(diagnosis.evidence)}\n"
                    f"IMPACT CENTER: {impact.center}\n"
                    f"SOURCE SLICES: {impact.source_context}\n"
                    f"PRIOR FIX MEMORY: {[record.patch_summary for record in history if record.success]}"
                ),
            ),
        ]
        return tuple(compact)

    def propose_repair(self, traceback_text: str) -> AgentRuntimeResult:
        diagnosis = self.diagnose_failure(traceback_text)
        context: Sequence[ChatMessage] = self.repair_context(traceback_text)
        return self.agent_runtime.run(
            objective=(
                f"Repair {diagnosis.error_type} in {diagnosis.primary_frame.path if diagnosis.primary_frame else 'unknown'} "
                f"with minimal change. Root-cause hypothesis: {diagnosis.root_cause_hypothesis}"
            ),
            task_kind="coding",
            context=context,
        )

    def record_repair(
        self,
        diagnosis: CausalDiagnosis,
        *,
        patch_summary: str,
        validation: str,
        success: bool,
    ) -> None:
        self.repair_memory.append(
            RepairMemoryRecord(
                error_type=diagnosis.error_type,
                signature=f"{diagnosis.error_type}:{diagnosis.message.casefold()}",
                root_cause=diagnosis.root_cause_hypothesis,
                patch_summary=patch_summary,
                validation=validation,
                success=success,
                timestamp=datetime.now(timezone.utc).isoformat(),
                evidence=diagnosis.evidence,
            )
        )
