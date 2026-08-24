from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .agent_graph import ExecutionBudget, JudgeVerdict, RecursiveAgentGraph, TaskNode, WorkArtifact
from .capability_benchmark import BenchmarkCase, CapabilityBenchmark
from .code_intelligence import CodeIntelligenceIndex
from .cognitive import CognitiveCore, VerificationResult
from .cognitive_state import Belief, CognitiveState, MemoryItem, MemoryKind
from .context_contract import build_cognitive_context
from .evolution import EvolutionBenchmark, EvolutionMemory, EvolutionRecord
from .fault_lab import FaultInjectionLab
from .memory_lifecycle import MemoryLifecycleManager
from .model_router import AdaptiveModelRouter, RoutingRequest
from .repository_twin import CausalRepositoryTwin
from .runtime_trace import RuntimeTraceGraph, TraceEvent
from .self_healing_runtime import FailureSignatureStore, SelfHealingRuntime
from .stability import CodeHealth, CodeHealthStore


@dataclass(frozen=True)
class BenchmarkRun:
    results: dict[str, tuple[float, tuple[str, ...]]]
    elapsed_ms: float


def run_architecture_benchmark(
    *,
    benchmark: CapabilityBenchmark | None = None,
    cognitive: CognitiveCore | None = None,
    router: AdaptiveModelRouter | None = None,
    repository_root: str = ".",
) -> BenchmarkRun:
    """Run deterministic architecture checks and return execution evidence."""
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
    record("frontier-routing", decision.tier == "frontier" and decision.allow_parallel, f"tier={decision.tier};parallel={decision.allow_parallel}")

    plan = cognitive.plan("reason about a complex architecture", 2)
    record("recursive-agent-graph", bool(plan.steps), f"plan_kind={plan.kind};steps={len(plan.steps)}")

    graph = RecursiveAgentGraph(ExecutionBudget(max_depth=2, max_nodes=4, max_parallel=2, max_retries=1))

    def decompose(node: TaskNode, budget: ExecutionBudget):
        return () if node.depth > 0 else (TaskNode("child", "verify architecture fact", "research", 1, node.id),)

    def worker(node: TaskNode, children):
        return WorkArtifact(node.id, node.role, "verified", 0.9)

    def judge(node: TaskNode, artifact: WorkArtifact, children):
        return JudgeVerdict(True, 0.9)

    artifact = graph.run(TaskNode("root", "architecture smoke"), decompose=decompose, worker=worker, judge=judge)
    record("recursive-agent-graph", artifact.confidence >= 0.9, f"confidence={artifact.confidence:.2f}")

    state = CognitiveState(goal="benchmark memory")
    lifecycle = MemoryLifecycleManager()
    episodes = [
        MemoryItem(content="Validated repair strategy for database timeout", kind=MemoryKind.EPISODIC, importance=0.8, confidence=0.9, provenance=("benchmark",), tags=("coding", "repair")),
        MemoryItem(content="Validated repair strategy for database timeout", kind=MemoryKind.EPISODIC, importance=0.85, confidence=0.95, provenance=("benchmark-repeat",), tags=("coding", "repair")),
    ]
    for item in episodes:
        state.add_memory(item)
    promotions = lifecycle.consolidate(state.memories)
    context = build_cognitive_context(state, query="database timeout repair", memory_kinds=(MemoryKind.PROCEDURAL, MemoryKind.EPISODIC), memory_limit=4)
    record("cross-episode-repair-memory", bool(promotions) and promotions[0].kind == MemoryKind.PROCEDURAL and bool(context.memories), f"promotions={len(promotions)};context_memories={len(context.memories)}")

    verification: VerificationResult = cognitive.verify("A complete, evidence-backed response.", 2)
    record("trace-verification", verification.passed and verification.score >= 0.8, f"passed={verification.passed};score={verification.score:.2f}")

    index = CodeIntelligenceIndex()
    index.index_tree(repository_root)
    twin = CausalRepositoryTwin(index)
    twin.rebuild(repository_root)
    slice_result = twin.impact_slice("AIEngine", limit=5)
    record("program-graph-localization", bool(slice_result.nodes) and bool(slice_result.source_context), f"nodes={len(slice_result.nodes)};edges={len(slice_result.edges)}")

    trace = RuntimeTraceGraph()
    trace.add(TraceEvent("benchmark-cause", "2026-01-01T00:00:00+00:00", "state", "src/myai/engine.py", 1, "AIEngine", "state changed"))
    trace.add(TraceEvent("benchmark-error", "2026-01-01T00:00:01+00:00", "exception", "src/myai/engine.py", 2, "AIEngine", "TypeError: benchmark failure", parent_id="benchmark-cause"))
    trace.link("benchmark-cause", "benchmark-error", "caused")
    neighborhood = trace.neighborhood("benchmark-error")
    affected = twin.affected_files("src/myai/engine.py")
    record("causal-trace-diagnosis", bool(neighborhood) and any(event.kind == "exception" for event in neighborhood) and bool(affected), f"trace_events={len(neighborhood)};affected_files={len(affected)}")

    signatures = FailureSignatureStore("/tmp/myai-v10-failure-signatures.jsonl")
    healing = SelfHealingRuntime(episode_path="/tmp/myai-v10-repair-episodes.jsonl", signature_store=signatures)
    signature = healing.signature("TypeError", "database timeout 123", "AIEngine.generate")
    episode = healing.verified_repair(
        signature=signature,
        reproduce=lambda: (True, "synthetic reproduction"),
        validate=lambda: True,
        lesson="verified bounded recovery",
    )
    record("self-healing-runtime", episode.status == "verified" and bool(signatures.similar(signature)), f"status={episode.status};attempts={episode.attempts}")

    state_map = {"healthy": True}
    fault = FaultInjectionLab()
    fault_result = fault.run(fault.simple_toggle(state_map, "healthy"), detector=lambda: not state_map["healthy"])
    record("fault-injection-recovery", fault_result.detected and fault_result.recovered and fault_result.verified, f"detected={fault_result.detected};recovered={fault_result.recovered};verified={fault_result.verified}")

    health_store = CodeHealthStore("/tmp/myai-v10-code-health.json")
    health_store.upsert(CodeHealth("stable_symbol", 0.99, 0.99, 0.0, 0.01, 0.05, "2026-08-25T00:00:00+00:00"))
    record("stable-code-inspection", health_store.inspection_mode("stable_symbol") == "reuse", f"mode={health_store.inspection_mode('stable_symbol')}")

    state.observe("observation-a")
    for index_value in range(20):
        state.add_belief(Belief(statement=f"benchmark-belief-{index_value}", confidence=0.5))
    bounded = build_cognitive_context(state, query="database timeout", belief_limit=4, observation_limit=2, memory_limit=4)
    raw_size = len(state.summary())
    bounded_size = len(bounded.render())
    record("targeted-context", bounded_size < raw_size, f"raw_chars={raw_size};bounded_chars={bounded_size}")

    evolution = EvolutionMemory()
    evolution_benchmark = EvolutionBenchmark(evolution)
    baseline = EvolutionRecord("baseline-task", "baseline", True, 0.80, latency_ms=100.0)
    candidate = EvolutionRecord("candidate-task", "candidate", True, 0.90, latency_ms=110.0)
    promoted = evolution_benchmark.should_promote(candidate, baseline, min_delta=0.05)
    rejected = not evolution_benchmark.should_promote(EvolutionRecord("bad", "bad", False, 0.99, latency_ms=10.0), baseline, min_delta=0.05)
    record("strategy-promotion", promoted and rejected, f"promoted={promoted};rejected_bad={rejected}")

    elapsed_ms = (perf_counter() - started) * 1000.0
    for case_name in {case.name for case in benchmark.cases} - set(results):
        results[case_name] = (0.0, ("unmeasured-by-runner",))
    return BenchmarkRun(results=results, elapsed_ms=elapsed_ms)


def benchmark_cases(benchmark: CapabilityBenchmark | None = None) -> tuple[BenchmarkCase, ...]:
    return (benchmark or CapabilityBenchmark()).cases
