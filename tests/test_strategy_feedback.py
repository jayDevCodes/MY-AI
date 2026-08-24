from myai.context_contract import build_cognitive_context
from myai.cognitive_state import CognitiveState
from myai.evolution import EvolutionMemory, EvolutionRecord


def test_verified_strategy_is_ranked_for_future_use(tmp_path) -> None:
    memory = EvolutionMemory(tmp_path / "evolution.jsonl")
    memory.append(
        EvolutionRecord(
            task_id="task-a",
            strategy="route:balanced:coding",
            success=True,
            score=0.95,
            latency_ms=120.0,
        )
    )
    memory.append(
        EvolutionRecord(
            task_id="task-b",
            strategy="route:balanced:coding",
            success=True,
            score=0.90,
            latency_ms=130.0,
        )
    )
    ranked = memory.rank(limit=1)

    assert ranked[0].strategy == "route:balanced:coding"
    assert ranked[0].success_rate == 1.0
    assert ranked[0].mean_score > 0.9


def test_failed_strategy_does_not_beat_verified_strategy(tmp_path) -> None:
    memory = EvolutionMemory(tmp_path / "evolution.jsonl")
    memory.append(EvolutionRecord("good", "verified", True, 0.9))
    memory.append(EvolutionRecord("bad", "failed", False, 0.95))

    assert memory.best_strategy() == "verified"


def test_context_renders_proven_strategy_as_historical_signal() -> None:
    context = build_cognitive_context(
        CognitiveState(goal="debug service"),
        strategy_hints=("route:balanced:coding (success=1.00, score=0.93)",),
    )

    assert context.strategy_hints == ("route:balanced:coding (success=1.00, score=0.93)",)
    assert "PROVEN STRATEGIES" in context.render()
