from __future__ import annotations

from dataclasses import dataclass

from .cognitive_state import CognitiveState, MemoryItem, MemoryKind


@dataclass(frozen=True)
class CognitiveContext:
    """Bounded, task-aware projection of CognitiveState for model prompts."""

    goal: str | None
    subgoals: tuple[str, ...]
    strategy: str | None
    strategy_hints: tuple[str, ...]
    uncertainty: float
    observations: tuple[str, ...]
    beliefs: tuple[str, ...]
    memories: tuple[str, ...]
    capabilities: tuple[tuple[str, float], ...]
    world_snapshot: tuple[tuple[str, object], ...]

    def render(self) -> str:
        lines = [
            f"GOAL: {self.goal or '(none)'}",
            f"SUBGOALS: {', '.join(self.subgoals) or '(none)'}",
            f"STRATEGY: {self.strategy or '(none)'}",
            f"PROVEN STRATEGIES: {' | '.join(self.strategy_hints) or '(none)'}",
            f"UNCERTAINTY: {self.uncertainty:.2f}",
            f"OBSERVATIONS: {' | '.join(self.observations) or '(none)'}",
            f"BELIEFS: {' | '.join(self.beliefs) or '(none)'}",
            f"MEMORIES: {' | '.join(self.memories) or '(none)'}",
        ]
        if self.capabilities:
            lines.append("CAPABILITIES: " + ", ".join(f"{name}={score:.2f}" for name, score in self.capabilities))
        if self.world_snapshot:
            lines.append("WORLD SNAPSHOT: " + "; ".join(f"{key}={value!r}" for key, value in self.world_snapshot))
        return "\n".join(lines)


def build_cognitive_context(
    state: CognitiveState,
    *,
    query: str = "",
    memory_kinds: tuple[MemoryKind, ...] = (),
    memory_limit: int = 6,
    belief_limit: int = 6,
    observation_limit: int = 6,
    capability_limit: int = 8,
    world_limit: int = 8,
    strategy_hints: tuple[str, ...] = (),
) -> CognitiveContext:
    memories: list[MemoryItem] = []
    selected_kinds = memory_kinds or tuple(MemoryKind)
    for kind in selected_kinds:
        remaining = max(0, memory_limit - len(memories))
        if remaining == 0:
            break
        memories.extend(state.relevant_memories(query=query, kind=kind, limit=remaining))

    beliefs = tuple(
        belief.statement
        for belief in sorted(state.beliefs, key=lambda item: item.confidence, reverse=True)[:belief_limit]
    )
    observations = tuple(state.observations[-max(0, observation_limit):])
    capabilities = tuple(
        sorted(state.capability_snapshot.items(), key=lambda item: item[1], reverse=True)[:capability_limit]
    )
    world_items = tuple(list(state.world_snapshot.items())[:world_limit])

    return CognitiveContext(
        goal=state.goal,
        subgoals=tuple(state.subgoals),
        strategy=state.active_strategy,
        strategy_hints=strategy_hints[:6],
        uncertainty=state.uncertainty,
        observations=observations,
        beliefs=beliefs,
        memories=tuple(memory.content for memory in memories),
        capabilities=capabilities,
        world_snapshot=world_items,
    )
