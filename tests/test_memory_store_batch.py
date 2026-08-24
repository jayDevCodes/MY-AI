from pathlib import Path

from myai.cognitive_state import Belief, CognitiveState, MemoryItem, MemoryKind
from myai.memory_store import CognitiveMemoryStore


def test_persist_state_round_trips_multiple_items(tmp_path: Path) -> None:
    store = CognitiveMemoryStore(tmp_path / "memory.sqlite3")
    state = CognitiveState()
    state.add_memory(MemoryItem("memory-a", MemoryKind.SEMANTIC, importance=0.7, confidence=0.8))
    state.add_memory(MemoryItem("memory-b", MemoryKind.FAILURE, importance=0.9, confidence=0.95))
    state.add_belief(Belief("state is durable", confidence=0.9))

    store.persist_state(state)

    restored = CognitiveState()
    store.hydrate(restored)

    assert len(restored.memories) == 2
    assert restored.relevant_memories(limit=1)[0].content == "memory-b"
    assert restored.beliefs[0].statement == "state is durable"


def test_persist_state_skips_already_persisted_items(tmp_path: Path) -> None:
    store = CognitiveMemoryStore(tmp_path / "memory.sqlite3")
    state = CognitiveState()
    state.add_memory(MemoryItem("memory-a", MemoryKind.SEMANTIC))
    store.persist_state(state)

    state.add_memory(MemoryItem("memory-b", MemoryKind.PROCEDURAL))
    store.persist_state(state)
    store.persist_state(state)

    restored = CognitiveState()
    store.hydrate(restored)
    assert len(restored.memories) == 2
