from __future__ import annotations

from hashlib import sha256

from .capability_benchmark import CapabilityBenchmark, CapabilitySnapshot
from .capability_ledger import CapabilityLedger
from .evolution import EvolutionMemory, EvolutionRecord, StrategyScore
from .graph_v9 import ProgramGraph, ProgramSlice
from .runtime_trace import RuntimeTraceGraph
from .schemas import ChatMessage
from .v8_engine import V8AIEngine


class V9AIEngine(V8AIEngine):
    """V9.1 cognitive mesh with shared cognitive state and state-aware agents."""

    version = "v9.1"

    def __init__(self) -> None:
        super().__init__()
        self.program_graph = ProgramGraph(self.code_index)
        self.runtime_traces = RuntimeTraceGraph.load(self.settings.runtime_trace_path)
        self.evolution_memory = EvolutionMemory(self.settings.evolution_memory_path)
        self.capability_ledger = CapabilityLedger(self.settings.capability_ledger_path)
        self.capability_benchmark = CapabilityBenchmark()
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
        return self.capability_ledger.record(snapshot)
