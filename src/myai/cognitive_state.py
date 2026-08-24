from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


class MemoryKind(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    STRATEGIC = "strategic"
    FAILURE = "failure"


@dataclass(frozen=True)
class Belief:
    statement: str
    confidence: float = 0.5
    provenance: tuple[str, ...] = ()
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("belief confidence must be between 0 and 1")


@dataclass(frozen=True)
class MemoryItem:
    content: str
    kind: MemoryKind
    importance: float = 0.5
    confidence: float = 0.5
    provenance: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("memory content must not be empty")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("memory importance must be between 0 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("memory confidence must be between 0 and 1")


@dataclass
class CognitiveState:
    """Structured state shared across planning, memory, tools and verification."""

    goal: str | None = None
    subgoals: list[str] = field(default_factory=list)
    beliefs: list[Belief] = field(default_factory=list)
    memories: list[MemoryItem] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    active_strategy: str | None = None
    uncertainty: float = 0.5
    capability_snapshot: dict[str, float] = field(default_factory=dict)
    world_snapshot: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be between 0 and 1")

    def set_goal(self, goal: str, *, subgoals: list[str] | None = None) -> None:
        if not goal.strip():
            raise ValueError("goal must not be empty")
        self.goal = goal
        self.subgoals = list(subgoals or [])

    def add_belief(self, belief: Belief) -> None:
        self.beliefs.append(belief)

    def add_memory(self, memory: MemoryItem) -> None:
        self.memories.append(memory)

    def observe(self, observation: str) -> None:
        if observation.strip():
            self.observations.append(observation)

    def set_uncertainty(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("uncertainty must be between 0 and 1")
        self.uncertainty = value

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {term for term in re.findall(r"[a-z0-9_]{3,}", text.casefold())}

    def relevant_memories(
        self,
        *,
        query: str = "",
        kind: MemoryKind | None = None,
        limit: int = 8,
    ) -> tuple[MemoryItem, ...]:
        if limit <= 0:
            return ()
        items = [item for item in self.memories if kind is None or item.kind == kind]
        query_terms = self._terms(query)

        def score(item: MemoryItem) -> tuple[float, str]:
            base = item.importance * item.confidence
            if not query_terms:
                return base, item.created_at
            item_terms = self._terms(f"{item.content} {' '.join(item.tags)}")
            overlap = len(query_terms & item_terms) / max(1, len(query_terms))
            return base * (1.0 + 1.5 * overlap), item.created_at

        items.sort(key=score, reverse=True)
        return tuple(items[:limit])

    def summary(self) -> dict[str, object]:
        return {
            "goal": self.goal,
            "subgoals": tuple(self.subgoals[:8]),
            "beliefs": tuple(self.beliefs[:8]),
            "constraints": tuple(self.constraints[:8]),
            "observations": tuple(self.observations[-8:]),
            "active_strategy": self.active_strategy,
            "uncertainty": self.uncertainty,
            "capability_snapshot": dict(self.capability_snapshot),
            "world_snapshot": dict(list(self.world_snapshot.items())[:8]),
            "memory_count": len(self.memories),
        }
