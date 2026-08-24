from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class EvolutionRecord:
    task_id: str
    strategy: str
    success: bool
    score: float
    latency_ms: float = 0.0
    tokens: int = 0
    lessons: tuple[str, ...] = ()
    timestamp: str = ""


@dataclass(frozen=True)
class StrategyScore:
    strategy: str
    samples: int
    success_rate: float
    mean_score: float
    mean_latency_ms: float
    mean_tokens: float


class EvolutionMemory:
    """Compact execution memory used to improve future strategy selection."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[EvolutionRecord] = []
        if self.path and self.path.exists():
            self._load()

    def append(self, record: EvolutionRecord) -> None:
        self.records.append(record)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def rank(self, limit: int = 5) -> tuple[StrategyScore, ...]:
        grouped: dict[str, list[EvolutionRecord]] = {}
        for record in self.records:
            grouped.setdefault(record.strategy, []).append(record)
        scores = [
            StrategyScore(
                strategy=strategy,
                samples=len(items),
                success_rate=sum(item.success for item in items) / len(items),
                mean_score=mean(item.score for item in items),
                mean_latency_ms=mean(item.latency_ms for item in items),
                mean_tokens=mean(item.tokens for item in items),
            )
            for strategy, items in grouped.items()
        ]
        scores.sort(key=lambda item: (-item.success_rate, -item.mean_score, item.mean_latency_ms, item.mean_tokens))
        return tuple(scores[:limit])

    def best_strategy(self) -> str | None:
        ranked = self.rank(limit=1)
        return ranked[0].strategy if ranked else None

    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                self.records.append(
                    EvolutionRecord(
                        task_id=str(raw["task_id"]),
                        strategy=str(raw["strategy"]),
                        success=bool(raw["success"]),
                        score=float(raw["score"]),
                        latency_ms=float(raw.get("latency_ms", 0.0)),
                        tokens=int(raw.get("tokens", 0)),
                        lessons=tuple(str(value) for value in raw.get("lessons", [])),
                        timestamp=str(raw.get("timestamp", "")),
                    )
                )
            except (ValueError, TypeError, KeyError):
                continue


class EvolutionBenchmark:
    """Evaluate strategies on repeated tasks without allowing memory to replace evidence."""

    def __init__(self, memory: EvolutionMemory) -> None:
        self.memory = memory

    def compare(self, task_id: str, outcomes: tuple[EvolutionRecord, ...]) -> StrategyScore | None:
        for outcome in outcomes:
            if outcome.task_id != task_id:
                continue
            self.memory.append(outcome)
        ranked = self.memory.rank(limit=10)
        return ranked[0] if ranked else None

    def should_promote(self, candidate: EvolutionRecord, baseline: EvolutionRecord, min_delta: float = 0.05) -> bool:
        if not candidate.success:
            return False
        return candidate.score >= baseline.score + min_delta and candidate.latency_ms <= baseline.latency_ms * 1.5
