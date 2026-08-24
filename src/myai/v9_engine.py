from __future__ import annotations

from .evolution import EvolutionMemory, StrategyScore
from .graph_v9 import ProgramGraph, ProgramSlice
from .runtime_trace import RuntimeTraceGraph
from .v8_engine import V8AIEngine


class V9AIEngine(V8AIEngine):
    """V9 cognitive mesh over the V8 causal repository twin."""

    version = "v9.0"

    def __init__(self) -> None:
        super().__init__()
        self.program_graph = ProgramGraph(self.code_index)
        self.runtime_traces = RuntimeTraceGraph.load(self.settings.runtime_trace_path)
        self.evolution_memory = EvolutionMemory(self.settings.evolution_memory_path)
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
