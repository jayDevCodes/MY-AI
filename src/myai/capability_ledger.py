from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "reasoning",
    "coding",
    "repository_understanding",
    "debugging",
    "memory",
    "tool_use",
    "planning",
    "verification",
    "efficiency",
    "self_improvement",
)


@dataclass(frozen=True)
class CapabilityScore:
    dimension: str
    score: float
    status: str = "measured"
    evidence: tuple[str, ...] = ()
    known_limits: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilitySnapshot:
    version: str
    commit: str
    captured_at: str
    scores: tuple[CapabilityScore, ...]
    benchmark_count: int = 0
    benchmark_passes: int = 0
    regression_count: int = 0
    notes: tuple[str, ...] = ()

    def score(self, dimension: str) -> float | None:
        for item in self.scores:
            if item.dimension == dimension:
                return item.score
        return None


class CapabilityLedger:
    """Persistent, machine-readable capability state used as the next-upgrade baseline."""

    schema_version = 1

    def __init__(self, path: str | Path = "data/capability_ledger.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots: list[CapabilitySnapshot] = []
        self._load()

    @property
    def current(self) -> CapabilitySnapshot | None:
        return self.snapshots[-1] if self.snapshots else None

    def record(self, snapshot: CapabilitySnapshot) -> CapabilitySnapshot:
        self.snapshots.append(snapshot)
        payload = {
            "schema_version": self.schema_version,
            "history": [asdict(item) for item in self.snapshots],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return snapshot

    def delta(self, current: CapabilitySnapshot, previous: CapabilitySnapshot | None = None) -> dict[str, float]:
        baseline = previous or self.current
        if baseline is None:
            return {item.dimension: item.score for item in current.scores}
        old = {item.dimension: item.score for item in baseline.scores}
        return {
            item.dimension: round(item.score - old.get(item.dimension, 0.0), 4)
            for item in current.scores
        }

    def regressions(self, current: CapabilitySnapshot, previous: CapabilitySnapshot | None = None) -> tuple[str, ...]:
        return tuple(
            dimension
            for dimension, change in self.delta(current, previous).items()
            if change < -0.02
        )

    def baseline(self) -> dict[str, Any]:
        current = self.current
        if current is None:
            return {"version": None, "commit": None, "scores": {}, "regressions": []}
        return {
            "version": current.version,
            "commit": current.commit,
            "captured_at": current.captured_at,
            "scores": {item.dimension: item.score for item in current.scores},
            "regressions": list(self.regressions(current, self.snapshots[-2] if len(self.snapshots) > 1 else None)),
            "benchmark_count": current.benchmark_count,
            "benchmark_passes": current.benchmark_passes,
        }

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for item in raw.get("history", []):
                scores = tuple(
                    CapabilityScore(
                        dimension=str(score["dimension"]),
                        score=float(score["score"]),
                        status=str(score.get("status", "measured")),
                        evidence=tuple(str(value) for value in score.get("evidence", [])),
                        known_limits=tuple(str(value) for value in score.get("known_limits", [])),
                    )
                    for score in item.get("scores", [])
                )
                self.snapshots.append(
                    CapabilitySnapshot(
                        version=str(item["version"]),
                        commit=str(item["commit"]),
                        captured_at=str(item["captured_at"]),
                        scores=scores,
                        benchmark_count=int(item.get("benchmark_count", 0)),
                        benchmark_passes=int(item.get("benchmark_passes", 0)),
                        regression_count=int(item.get("regression_count", 0)),
                        notes=tuple(str(value) for value in item.get("notes", [])),
                    )
                )
        except (OSError, ValueError, TypeError, KeyError):
            self.snapshots.clear()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
