from pathlib import Path

from myai.code_intelligence import CodeIntelligenceIndex
from myai.evolution import EvolutionBenchmark, EvolutionMemory, EvolutionRecord
from myai.graph_v9 import ProgramGraph
from myai.runtime_trace import RuntimeTraceGraph, TraceEvent


def test_program_graph_builds_calls_data_and_control(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "def helper(value):\n"
        "    return value\n\n"
        "def main(value):\n"
        "    if value:\n"
        "        return helper(value)\n"
        "    return 0\n",
        encoding="utf-8",
    )
    index = CodeIntelligenceIndex()
    assert index.index_tree(tmp_path) == 1
    graph = ProgramGraph(index)
    graph.build(tmp_path)
    kinds = {edge.kind for edge in graph.edges}
    assert "calls" in kinds
    assert "control_flow" in kinds

    result = graph.slice("main")
    assert result.nodes
    assert result.source_context


def test_runtime_trace_graph_persists_events_and_links(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    graph = RuntimeTraceGraph(path)
    graph.add(TraceEvent("e1", "2026-01-01T00:00:00Z", "exception", "sample.py", 4, "main", "ValueError: bad"))
    graph.add(TraceEvent("e2", "2026-01-01T00:00:01Z", "state", "sample.py", 3, "helper", "value=None"))
    graph.link("e2", "e1", "precedes")
    restored = RuntimeTraceGraph.load(path)
    assert restored.events["e1"].kind == "exception"
    assert len(restored.neighborhood("e1")) == 2


def test_evolution_memory_ranks_strategies_and_gates_promotion(tmp_path: Path) -> None:
    memory = EvolutionMemory(tmp_path / "evolution.jsonl")
    benchmark = EvolutionBenchmark(memory)
    baseline = EvolutionRecord("task-1", "baseline", True, 0.70, latency_ms=100, tokens=1000)
    candidate = EvolutionRecord("task-1", "graph-slice", True, 0.82, latency_ms=110, tokens=700)
    best = benchmark.compare("task-1", (baseline, candidate))
    assert best is not None
    assert best.strategy == "graph-slice"
    assert benchmark.should_promote(candidate, baseline) is True
