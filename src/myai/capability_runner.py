from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .agent_graph import ExecutionBudget, JudgeVerdict, RecursiveAgentGraph, TaskNode, WorkArtifact
from .capability_benchmark import BenchmarkCase, CapabilityBenchmark
from .cognitive import CognitiveCore
from .model_router import AdaptiveModelRouter, RoutingRequest


@dataclass(frozen=True)
class BenchmarkRun:
    results: dict[str, tuple[float, tuple[str, ...]]]
    elapsed_ms: float


def run_architecture_benchmark(
    *,
    benchmark: CapabilityBenchmark | None = None,
    cognitive: CognitiveCore | None = None,
    router: AdaptiveModelRouter | None = None,
) -> BenchmarkRun:
    """Execute deterministic architecture-level checks without requiring a provider model."""
    benchmark = benchmark or CapabilityBenchmark()
    cognitive = cognitive or CognitiveCore()
    router = router or AdaptiveModelRouter()
    started = perf_counter()
    results: dict[str, tuple[float, tuple[str, ...]]] = {}

    def record(case_name: str, passed: bool, evidence: str) -> None:
        existing = results.get(case_name)
        score = 1.0 if passed else 0.0
        if existing is None:
            results[case_name] = (score, (evidence,))
        else:
            results[case_name] = ((existing[0] + score) / 2.0, existing[1] + (evidence,))

    decision = router.choose(RoutingRequest(task_kind="research", complexity=0.9, uncertainty=0.9))
    record(
        "frontier-routing",
        decision.tier == "frontier" and decision.allow_parallel,
        f"tier={decision.tier};parallel={decision.allow_parallel}",
    )

    plan = cognitive.plan("reason about a complex architecture", 2)
    record(
        "recursive-agent-graph",
        bool(plan.steps),
        f"plan_kind={plan.kind};steps={len(plan.steps)}",
    )

    graph = RecursiveAgentGraph(
        ExecutionBudget(max_depth=2, max_nodes=4, max_parallel=2, max_retries=1)
    )

    def decompose(node: TaskNode, budget: ExecutionBudget):
        return () if node.depth > 0 else (
            TaskNode("child", "verify architecture fact", "research", 1, node.id),
        )

    def worker(node: TaskNode, children):
        return WorkArtifact(node.id, node.role, "verified", 0.9)

    def judge(node: TaskNode, artifact: WorkArtifact, children):
        return JudgeVerdict(True, 0.9)

    artifact = graph.run(
        TaskNode("root", "architecture smoke"),
        decompose=decompose,
        worker=worker,
        judge=judge,
    )
    record(
        "recursive-agent-graph",
        artifact.confidence >= 0.9,
        f"confidence={artifact.confidence:.2f}",
    )

    return BenchmarkRun(
        results=results,
        elapsed_ms=(perf_counter() - started) * 1000.0,
    )


def benchmark_cases(benchmark: CapabilityBenchmark | None = None) -> tuple[BenchmarkCase, ...]:
    return (benchmark or CapabilityBenchmark()).cases
