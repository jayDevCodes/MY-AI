from myai.context_contract import build_cognitive_context
from myai.cognitive_state import Belief, CognitiveState, MemoryItem, MemoryKind


def test_cognitive_context_is_bounded_and_query_aware() -> None:
    state = CognitiveState(goal="debug auth")
    state.add_belief(Belief("auth uses refresh tokens", confidence=0.9))
    state.add_belief(Belief("unrelated historical belief", confidence=1.0))
    state.add_memory(
        MemoryItem("refresh token timeout caused auth failure", MemoryKind.FAILURE, importance=0.8, confidence=0.9, tags=("auth",))
    )
    for index in range(12):
        state.add_memory(
            MemoryItem(f"unrelated memory {index}", MemoryKind.EPISODIC, importance=0.7, confidence=0.7)
        )
    state.observe("verification passed")

    context = build_cognitive_context(state, query="auth refresh token", memory_limit=2, belief_limit=1, observation_limit=1)

    assert len(context.memories) <= 2
    assert "refresh token timeout caused auth failure" in context.memories
    assert len(context.beliefs) == 1
    assert len(context.observations) == 1
    assert "auth" in context.render().lower()


def test_relevant_memories_can_rank_by_query_overlap() -> None:
    state = CognitiveState()
    target = MemoryItem("database connection pool timeout", MemoryKind.PROCEDURAL, importance=0.5, confidence=0.5)
    distractor = MemoryItem("frontend css layout hint", MemoryKind.PROCEDURAL, importance=0.9, confidence=0.9)
    state.add_memory(distractor)
    state.add_memory(target)

    memories = state.relevant_memories(query="database timeout", limit=1)

    assert memories[0].content == target.content


def test_context_preserves_proven_strategy_hints_as_historical_signals() -> None:
    state = CognitiveState(goal="debug service")
    context = build_cognitive_context(
        state,
        strategy_hints=("route:balanced:coding (success=1.00, score=0.93)",),
    )

    assert context.strategy_hints == ("route:balanced:coding (success=1.00, score=0.93)",)
    assert "PROVEN STRATEGIES" in context.render()
