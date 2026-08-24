from pathlib import Path

from myai.cognitive_state import Belief, CognitiveState, MemoryItem, MemoryKind
from myai.memory_store import CognitiveMemoryStore


def test_memory_store_round_trip(tmp_path: Path) -> None:
    store = CognitiveMemoryStore(tmp_path / "memory.sqlite3")
    state = CognitiveState()
    state.add_memory(
        MemoryItem(
            content="Verified timeout repair",
            kind=MemoryKind.FAILURE,
            importance=0.9,
            confidence=0.95,
            provenance=("test",),
            tags=("repair",),
        )
    )
    state.add_belief(Belief("The repair is safe", confidence=0.8, provenance=("test",)))
    store.persist_state(state)

    restored = CognitiveState()
    store.hydrate(restored)

    assert restored.relevant_memories(limit=1)[0].content == "Verified timeout repair"
    assert restored.beliefs[0].statement == "The repair is safe"


def test_memory_store_deduplicates_same_kind(tmp_path: Path) -> None:
    store = CognitiveMemoryStore(tmp_path / "memory.sqlite3")
    first = MemoryItem(content="same", kind=MemoryKind.SEMANTIC, importance=0.4, confidence=0.4)
    second = MemoryItem(content="same", kind=MemoryKind.SEMANTIC, importance=0.8, confidence=0.9)
    store.remember(first)
    store.remember(second)

    memories = store.load_memories()
    assert len(memories) == 1
    assert memories[0].importance == 0.8
    assert memories[0].confidence == 0.9
