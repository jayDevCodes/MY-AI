from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .capability_benchmark import CapabilityBenchmark
from .cognitive_compute import CognitiveComputeController
from .model_router import AdaptiveModelRouter, RoutingRequest


@dataclass(frozen=True)
class ModelReport:
    """Self-derived architecture/capability report for the running MY-AI engine."""

    version: str
    active_features: tuple[str, ...]
    capability_baseline: dict[str, Any]
    benchmark_cases: tuple[str, ...]
    strategy_summary: tuple[dict[str, Any], ...]
    compute_policies: tuple[dict[str, Any], ...]
    measured_claim_policy: str
    known_limits: tuple[str, ...]
    next_upgrade_targets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FEATURES = (
    ("program_graph", "program_graph"),
    ("repository_twin", "repository_twin"),
    ("runtime_traces", "runtime_traces"),
    ("evolution_memory", "evolution_memory"),
    ("capability_ledger", "capability_ledger"),
    ("capability_benchmark", "capability_benchmark"),
    ("memory_store", "memory_store"),
    ("memory_lifecycle", "memory_lifecycle"),
    ("cognitive_state", "cognitive_state"),
    ("agent_runtime", "agent_runtime"),
    ("model_router", "model_router"),
    ("repository_code_index", "code_index"),
    ("repair_memory", "repair_memory"),
)

_REPRESENTATIVE_TASKS = (
    RoutingRequest(task_kind="research", complexity=0.9, uncertainty=0.8, risk="high"),
    RoutingRequest(task_kind="coding", complexity=0.6, uncertainty=0.4, risk="medium"),
    RoutingRequest(task_kind="formatting", complexity=0.2, uncertainty=0.1, risk="low", latency_sensitive=True),
)


def build_model_report(engine: Any) -> ModelReport:
    """Derive a report from live engine state rather than hand-maintained claims."""
    features = tuple(label for label, attr in _FEATURES if hasattr(engine, attr))
    benchmark = getattr(engine, "capability_benchmark", CapabilityBenchmark())
    ledger = getattr(engine, "capability_ledger", None)
    baseline = ledger.baseline() if ledger is not None else {}

    strategy_scores = getattr(engine, "strategy_scores", lambda limit=5: ()) (limit=5)
    strategy_summary = tuple(
        {
            "strategy": item.strategy,
            "samples": item.samples,
            "success_rate": round(item.success_rate, 4),
            "mean_score": round(item.mean_score, 4),
            "mean_latency_ms": round(item.mean_latency_ms, 2),
            "mean_tokens": round(item.mean_tokens, 2),
        }
        for item in strategy_scores
    )

    router = AdaptiveModelRouter()
    controller = CognitiveComputeController(ledger=ledger, strategy_scores=tuple(strategy_scores))
    policies = []
    for request in _REPRESENTATIVE_TASKS:
        decision = router.choose(request)
        policy = controller.policy_for(request, decision)
        policies.append(
            {
                "task_kind": request.task_kind,
                "tier": decision.tier,
                "reasoning_depth": policy.reasoning_depth,
                "max_parallel": policy.max_parallel,
                "max_retries": policy.max_retries,
                "verification_passes": policy.verification_passes,
                "reason": policy.reason,
            }
        )

    return ModelReport(
        version=str(getattr(engine, "version", "unknown")),
        active_features=features,
        capability_baseline=baseline,
        benchmark_cases=tuple(case.name for case in benchmark.cases),
        strategy_summary=strategy_summary,
        compute_policies=tuple(policies),
        measured_claim_policy="Numeric capability claims require executable evidence; architecture presence alone is not a score.",
        known_limits=(
            "Provider-dependent tool use remains unmeasured by the deterministic report.",
            "Deterministic architecture evidence does not establish frontier-model task quality.",
            "Model-weight training is not part of the current self-evolution loop.",
        ),
        next_upgrade_targets=(
            "provider-backed end-to-end benchmark harness",
            "longitudinal repeated-run evaluation with confidence intervals",
            "real runtime instrumentation and automatic trace capture",
            "sandbox execution with rollback",
            "predictive world-state and counterfactual simulation",
        ),
    )
