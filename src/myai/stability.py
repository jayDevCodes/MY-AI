from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeHealth:
    symbol: str
    stability: float
    verification_confidence: float
    failure_rate: float
    change_frequency: float
    dependency_centrality: float
    last_verified: str | None = None

    @property
    def inspection_mode(self) -> str:
        score = min(
            1.0,
            0.3 * self.stability
            + 0.3 * self.verification_confidence
            + 0.15 * (1.0 - self.failure_rate)
            + 0.15 * (1.0 - self.change_frequency)
            + 0.10 * (1.0 - self.dependency_centrality),
        )
        if score >= 0.85:
            return "reuse"
        if score >= 0.60:
            return "targeted"
        return "deep"


class CodeHealthStore:
    """Persistent symbol health metadata used to avoid needless deep rereads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, CodeHealth] = {}
        self._load()

    def upsert(self, item: CodeHealth) -> None:
        self._items[item.symbol] = item
        self._persist()

    def get(self, symbol: str) -> CodeHealth | None:
        return self._items.get(symbol)

    def inspection_mode(self, symbol: str) -> str:
        item = self.get(symbol)
        return item.inspection_mode if item else "deep"

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for key, value in raw.items():
            try:
                self._items[key] = CodeHealth(**value)
            except (TypeError, ValueError):
                continue

    def _persist(self) -> None:
        self.path.write_text(
            json.dumps({key: asdict(value) for key, value in self._items.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
