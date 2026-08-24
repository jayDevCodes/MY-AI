from myai.capability_ledger import CapabilityLedger, CapabilityScore, CapabilitySnapshot
from myai.cognitive_compute import CognitiveComputeController
from myai.evolution import StrategyScore
from myai.model_router import AdaptiveModelRouter, RoutingRequest


def _ledger(tmp_path, score: float) -> CapabilityLedger:
    ledger = CapabilityLedger(tmp_path / "capabilities.json")
    scores = tuple(
        CapabilityScore(dimension, score, status="measured", evidence=("test-evidence",))
        for dimension in (
            "reasoning",
            "coding",
            "planning",
            "verification",
            "debugging",
            "memory",
        )
    )
    ledger.record(
        CapabilitySnapshot(
            version="v10",
            commit="test",
            captured_at="2026-08-24T00:00:00+00:00",
            scores=scores,
            benchmark_count=6,
            benchmark_passes=6,
        )
    )
    return ledger


def test_low_capability_gap_allocates_more_compute(tmp_path) -> None:
    ledger = _ledger(tmp_path, 0.10)
    request = RoutingRequest(task_kind="coding", complexity=0.5, uncertainty=0.2)
    decision = AdaptiveModelRouter().choose(request)

    policy = CognitiveComputeController(ledger=ledger).policy_for(request, decision)

    assert policy.reasoning_depth == 3
    assert policy.verification_passes == 2
    assert policy.execution_budget().max_depth == 3


def test_high_capability_and_reliable_strategy_can_reduce_parallelism(tmp_path) -> None:
    ledger = _ledger(tmp_path, 0.95)
    strategy = StrategyScore("coding", 8, 1.0, 0.95, 100.0, 100.0)
    request = RoutingRequest(task_kind="coding", complexity=0.6, uncertainty=0.1)
    decision = AdaptiveModelRouter().choose(request)

    policy = CognitiveComputeController(ledger=ledger, strategy_scores=(strategy,)).policy_for(request, decision)

    assert policy.reasoning_depth == 1
    assert policy.max_parallel == 1
    assert policy.verification_passes == 1
