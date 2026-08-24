from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Iterable

from .cognitive_state import MemoryItem, MemoryKind


@dataclass(frozen=True)
class MemoryLifecycleConfig:
    max_working: int = 16
    max_episodic: int = 96
    max_semantic: int = 128
    max_procedural: int = 96
    max_strategic: int = 64
    max_failure: int = 96
    promotion_min_recurrence: int = 2
    promotion_min_confidence: float = 0.75
    stale_after_days: float = 30.0


class MemoryLifecycleManager:
    """Deterministic, evidence-gated lifecycle policy for long-term memory.

    Raw episodic memories remain first-class evidence. Promotion creates a new
    typed memory; it never rewrites or deletes the source episodes implicitly.
    """

    def __init__(self, config: MemoryLifecycleConfig | None = None) -> None:
        self.config = config or MemoryLifecycleConfig()

    def retain(self, memories: Iterable[MemoryItem]) -> tuple[MemoryItem, ...]:
        items = list(memories)
        limits = {
            MemoryKind.WORKING: self.config.max_working,
            MemoryKind.EPISODIC: self.config.max_episodic,
            MemoryKind.SEMANTIC: self.config.max_semantic,
            MemoryKind.PROCEDURAL: self.config.max_procedural,
            MemoryKind.STRATEGIC: self.config.max_strategic,
            MemoryKind.FAILURE: self.config.max_failure,
        }
        retained: list[MemoryItem] = []
        for kind, limit in limits.items():
            bucket = [item for item in items if item.kind == kind]
            bucket.sort(key=lambda item: (self.decay_score(item), item.created_at), reverse=True)
            retained.extend(bucket[: max(0, limit)])
        return tuple(retained)

    def consolidate(self, memories: Iterable[MemoryItem]) -> tuple[MemoryItem, ...]:
        """Promote repeated, consistently high-confidence episodes without rewriting evidence."""
        items = list(memories)
        groups: dict[str, list[MemoryItem]] = {}
        for item in items:
            if item.kind != MemoryKind.EPISODIC:
                continue
            key = self._normalize(item.content)
            if key:
                groups.setdefault(key, []).append(item)

        promotions: list[MemoryItem] = []
        for episodes in groups.values():
            if len(episodes) < self.config.promotion_min_recurrence:
                continue
            if any(item.confidence < self.config.promotion_min_confidence for item in episodes):
                continue
            newest = max(episodes, key=lambda item: item.created_at)
            promotion_kind = (
                MemoryKind.PROCEDURAL
                if any("coding" in tag or "repair" in tag for item in episodes for tag in item.tags)
                else MemoryKind.SEMANTIC
            )
            promotions.append(
                MemoryItem(
                    content=newest.content,
                    kind=promotion_kind,
                    importance=max(item.importance for item in episodes),
                    confidence=min(item.confidence for item in episodes),
                    provenance=tuple(sorted({source for item in episodes for source in item.provenance})),
                    tags=tuple(sorted({tag for item in episodes for tag in item.tags} | {"consolidated", f"recurrence:{len(episodes)}"})),
                    created_at=newest.created_at,
                )
            )
        return tuple(promotions)

    def decay_score(self, memory: MemoryItem, *, now: datetime | None = None) -> float:
        """Return a bounded retention score with mild age decay."""
        now = now or datetime.now(timezone.utc)
        try:
            created = datetime.fromisoformat(memory.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - created).total_seconds() / 86400.0)
        except ValueError:
            age_days = self.config.stale_after_days
        decay = 0.5 ** (age_days / max(1.0, self.config.stale_after_days))
        kind_boost = 1.15 if memory.kind in {MemoryKind.PROCEDURAL, MemoryKind.STRATEGIC} else 1.0
        return max(0.0, min(1.0, memory.importance * memory.confidence * decay * kind_boost))

    @staticmethod
    def _normalize(content: str) -> str:
        return re.sub(r"\s+", " ", content.strip().casefold())
