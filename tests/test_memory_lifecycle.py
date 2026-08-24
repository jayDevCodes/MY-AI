from myai.cognitive_state import MemoryItem, MemoryKind
from myai.memory_lifecycle import MemoryLifecycleManager


def test_consolidation_requires_recurrence_and_consistent_confidence() -> None:
    manager = MemoryLifecycleManager()
    episode = MemoryItem(
        "The timeout repair passed integration tests",
        MemoryKind.EPISODIC,
        importance=0.8,
        confidence=0.9,
        provenance=("verification",),
        tags=("coding", "repair"),
    )
    low_confidence = MemoryItem(
        "The timeout repair passed integration tests",
        MemoryKind.EPISODIC,
        importance=0.8,
        confidence=0.4,
        provenance=("unverified",),
        tags=("coding",),
    )

    assert manager.consolidate((episode,)) == ()
    promoted = manager.consolidate((episode, episode))
    assert len(promoted) == 1
    assert promoted[0].kind == MemoryKind.PROCEDURAL
    assert promoted[0].confidence == episode.confidence
    assert manager.consolidate((episode, low_confidence)) == ()


def test_decay_prefers_recent_high_value_memory() -> None:
    manager = MemoryLifecycleManager()
    recent = MemoryItem("recent", MemoryKind.PROCEDURAL, importance=0.8, confidence=0.9)
    old = MemoryItem(
        "old",
        MemoryKind.EPISODIC,
        importance=0.8,
        confidence=0.9,
        created_at="2025-01-01T00:00:00+00:00",
    )

    assert manager.decay_score(recent) > manager.decay_score(old)
