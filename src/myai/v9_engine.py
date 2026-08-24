from __future__ import annotations

from hashlib import sha256

from .capability_benchmark import CapabilityBenchmark, CapabilitySnapshot
from .capability_ledger import CapabilityLedger
from .cognitive_compute import CognitiveComputeController, CognitiveComputePolicy
from .evolution import EvolutionMemory, EvolutionRecord, StrategyScore
from .fault_lab import FaultCase, FaultInjectionLab, FaultResult
from .graph_v9 import ProgramGraph, ProgramSlice
from .model_report import ModelReport, build_model_report
from .model_router import AdaptiveModelRouter, RoutingDecision, RoutingRequest
from .runtime_trace import RuntimeTraceGraph
from .schemas import ChatMessage
from .self_healing import CausalDiagnosis, CausalErrorEngine, RepairMemory, RepairMemoryRecord
from .self_healing_runtime import FailureSignature, FailureSignatureStore, RepairEpisode, SelfHealingRuntime
from .stability import CodeHealth, CodeHealthStore
from .v8_engine import V8AIEngine


class V9AIEngine(V8AIEngine):
    """V9.1 cognitive mesh with bounded self-healing runtime controls."""

    version = "v9.1"

    def __init__(self) -> None:
        super().__init__()
        self.program_graph = ProgramGraph(self.code_index)
        self.runtime_traces = RuntimeTraceGraph.load(self.settings.runtime_trace_path)
        self.evolution_memory = EvolutionMemory(self.settings.evolution_memory_path)
        self.capability_ledger = CapabilityLedger(self.settings.capability_ledger_path)
        self.capability_benchmark = CapabilityBenchmark()
        self.compute_controller = CognitiveComputeController(
            ledger=self.capability_ledger,
            strategy_scores=self.strategy_scores(limit=8),
        )
        self.failure_signatures = FailureSignatureStore(self.settings.failure_signature_path)
        self.self_healing_runtime = SelfHealingRuntime(
            episode_path=self.settings.repair_episode_path,
            signature_store=self.failure_signatures,
        )
        self.code_health = CodeHealthStore(self.settings.code_health_path)
        self.fault_lab = FaultInjectionLab()
        self._refresh_program_graph()

    def _refresh_program_graph(self) -> None:
        self._ensure_code_index_fresh()
        self.repository_twin.rebuild(self.settings.code_index_root)
        self.program_graph.build(self.settings.code_index_root)

    def refresh_cognitive_graph(self) -> None:
        self.refresh_repository_twin()
        self._refresh_program_graph()

    def program_slice(self, query: str, limit: int | None = None) -> ProgramSlice:
        self._refresh_program_graph()
        return self.program_graph.slice(query, limit=limit or self.settings.code_context_limit)

    def strategy_scores(self, limit: int = 5) -> tuple[StrategyScore, ...]:
        return self.evolution_memory.rank(limit=limit)

    def best_strategy(self) -> str | None:
        return self.evolution_memory.best_strategy()

    def compute_policy(self, request: RoutingRequest) -> CognitiveComputePolicy:
        """Use measured capability gaps and historical strategy evidence to size bounded compute."""
        decision = AdaptiveModelRouter().choose(request)
        controller = CognitiveComputeController(
            ledger=self.capability_ledger,
            strategy_scores=self.strategy_scores(limit=8),
        )
        return controller.policy_for(request, decision)

    def model_report(self) -> ModelReport:
        """Return a live report derived from this engine's current architecture and evidence."""
        return build_model_report(self)

    def record_generation_experience(
        self,
        *,
        task: str,
        strategy: str,
        success: bool,
        score: float,
        latency_ms: float,
    ) -> None:
        task_id = sha256(task.strip().encode("utf-8")).hexdigest()[:16]
        self.evolution_memory.append(
            EvolutionRecord(
                task_id=task_id,
                strategy=strategy,
                success=success,
                score=max(0.0, min(1.0, score)),
                latency_ms=max(0.0, latency_ms),
                lessons=("verified" if success else "verification-failed",),
            )
        )
        self.compute_controller = CognitiveComputeController(
            ledger=self.capability_ledger,
            strategy_scores=self.strategy_scores(limit=8),
        )

    def repair_context_v9(self, traceback_text: str) -> tuple[ChatMessage, ...]:
        diagnosis = self.diagnose_failure(traceback_text)
        query = diagnosis.primary_frame.symbol if diagnosis.primary_frame else diagnosis.message
        program = self.program_slice(query, limit=self.settings.code_context_limit)
        trace_events = ()
        if diagnosis.primary_frame:
            trace_events = tuple(
                event
                for event in self.runtime_traces.events.values()
                if event.path == diagnosis.primary_frame.path
                and (event.line is None or abs(event.line - diagnosis.primary_frame.line) <= 8)
            )
        strategy = self.best_strategy() or "baseline-targeted-repair"
        return (
            ChatMessage(
                role="system",
                content=(
                    "You are MY-AI V9.1 repair specialist. Use the program graph, runtime evidence, and shared cognitive state as hard context constraints. "
                    "Do not reread unrelated files. Preserve verified code and propose the smallest causally justified repair."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"ERROR: {diagnosis.error_type}: {diagnosis.message}\n"
                    f"ROOT CAUSE HYPOTHESIS: {diagnosis.root_cause_hypothesis}\n"
                    f"CONFIDENCE: {diagnosis.confidence:.2f}\n"
                    f"PROGRAM CENTER: {program.center}\n"
                    f"PROGRAM NODES: {program.nodes}\n"
                    f"PROGRAM EDGES: {program.edges}\n"
                    f"SOURCE CONTEXT: {program.source_context}\n"
                    f"RUNTIME NEIGHBORHOOD: {trace_events}\n"
                    f"BEST HISTORICAL STRATEGY: {strategy}"
                ),
            ),
        )

    def failure_signature(self, traceback_text: str) -> FailureSignature:
        diagnosis = self.diagnose_failure(traceback_text)
        symbol = diagnosis.primary_frame.symbol if diagnosis.primary_frame else ""
        return self.self_healing_runtime.signature(diagnosis.error_type, diagnosis.message, symbol)

    def inspection_mode(self, symbol: str) -> str:
        return self.code_health.inspection_mode(symbol)

    def record_code_health(self, health: CodeHealth) -> None:
        self.code_health.upsert(health)

    def run_repair_episode(
        self,
        traceback_text: str,
        *,
        reproduce,
        validate,
        lesson: str = "",
    ) -> RepairEpisode:
        """Record a bounded repair episode. Code promotion remains an external, verified step."""
        diagnosis = self.diagnose_failure(traceback_text)
        signature = self.self_healing_runtime.signature(
            diagnosis.error_type,
            diagnosis.message,
            diagnosis.primary_frame.symbol if diagnosis.primary_frame else "",
        )
        return self.self_healing_runtime.verified_repair(
            signature=signature,
            reproduce=reproduce,
            validate=validate,
            lesson=lesson,
        )

    def run_fault_test(self, case: FaultCase, *, detector) -> FaultResult:
        return self.fault_lab.run(case, detector=detector)

    def capability_baseline(self) -> dict[str, object]:
        return self.capability_ledger.baseline()

    def record_capability_snapshot(
        self,
        commit: str,
        results: dict[str, tuple[float, tuple[str, ...]]],
        notes: tuple[str, ...] = (),
    ) -> CapabilitySnapshot:
        snapshot = self.capability_benchmark.snapshot(
            version=self.version,
            commit=commit,
            results=results,
            notes=notes,
        )
        recorded = self.capability_ledger.record(snapshot)
        self.compute_controller = CognitiveComputeController(
            ledger=self.capability_ledger,
            strategy_scores=self.strategy_scores(limit=8),
        )
        return recorded
