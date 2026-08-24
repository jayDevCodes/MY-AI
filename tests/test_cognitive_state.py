from myai.cognitive_state import Belief, CognitiveState, MemoryItem, MemoryKind


def test_cognitive_state_tracks_goal_beliefs_and_observations() -> None:
    state = CognitiveState()
    state.set_goal("Build a verified repair")
    state.add_belief(Belief("The failure is localized", confidence=0.9, provenance=("trace",)))
    state.observe("pytest reported a failing assertion")

    assert state.goal == "Build a verified repair"
    assert state.beliefs[0].confidence == 0.9
    assert state.observations == ["pytest reported a failing assertion"]


def test_memory_is_ranked_by_importance_and_confidence() -> None:
    state = CognitiveState()
    state.add_memory(
        MemoryItem("weak memory", MemoryKind.EPISODIC, importance=0.9, confidence=0.2)
    )
    state.add_memory(
        MemoryItem("strong memory", MemoryKind.PROCEDURAL, importance=0.8, confidence=1.0)
    )

    ranked = state.relevant_memories(limit=2)
    assert [item.content for item in ranked] == ["strong memory", "weak memory"]


def test_memory_kind_filter() -> None:
    state = CognitiveState()
    state.add_memory(MemoryItem("fact", MemoryKind.SEMANTIC))
    state.add_memory(MemoryItem("lesson", MemoryKind.STRATEGIC))

    ranked = state.relevant_memories(kind=MemoryKind.STRATEGIC)
    assert [item.content for item in ranked] == ["lesson"]
