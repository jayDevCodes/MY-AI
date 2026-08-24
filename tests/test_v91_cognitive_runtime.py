from myai.cognitive_state import CognitiveState, MemoryItem, MemoryKind


def test_cognitive_state_tracks_goal_and_ranked_memory() -> None:
    state = CognitiveState()
    state.set_goal("debug authentication flow", subgoals=["locate failure", "verify repair"])
    state.add_memory(
        MemoryItem(
            content="Known token refresh failure",
            kind=MemoryKind.FAILURE,
            importance=0.9,
            confidence=0.8,
            provenance=("runtime-trace",),
        )
    )
    state.add_memory(
        MemoryItem(
            content="General coding hint",
            kind=MemoryKind.PROCEDURAL,
            importance=0.4,
            confidence=0.5,
        )
    )

    assert state.goal == "debug authentication flow"
    assert state.subgoals == ["locate failure", "verify repair"]
    assert state.relevant_memories(limit=1)[0].kind == MemoryKind.FAILURE
    assert state.summary()["memory_count"] == 2


def test_uncertainty_is_bounded() -> None:
    state = CognitiveState()
    state.set_uncertainty(0.2)
    assert state.uncertainty == 0.2
