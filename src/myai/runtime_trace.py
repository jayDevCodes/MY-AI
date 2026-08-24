from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    timestamp: str
    kind: str
    path: str | None = None
    line: int | None = None
    symbol: str | None = None
    message: str = ""
    data: tuple[tuple[str, str], ...] = ()
    parent_id: str | None = None


@dataclass(frozen=True)
class TraceCausalLink:
    source_id: str
    target_id: str
    relation: str


class RuntimeTraceGraph:
    """Persistent, compact runtime evidence graph for failures and repairs."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events: dict[str, TraceEvent] = {}
        self.links: set[TraceCausalLink] = set()

    def add(self, event: TraceEvent) -> None:
        self.events[event.event_id] = event
        self._persist(asdict(event))

    def link(self, source_id: str, target_id: str, relation: str) -> None:
        link = TraceCausalLink(source_id, target_id, relation)
        if link in self.links:
            return
        self.links.add(link)
        self._persist({"record_type": "link", **asdict(link)})

    def record_exception(self, event_id: str, path: str, line: int, symbol: str, exc: BaseException) -> TraceEvent:
        event = TraceEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            kind="exception",
            path=path,
            line=line,
            symbol=symbol,
            message=f"{type(exc).__name__}: {exc}",
        )
        self.add(event)
        return event

    def neighborhood(self, event_id: str, limit: int = 20) -> tuple[TraceEvent, ...]:
        related = {event_id}
        for link in self.links:
            if link.source_id == event_id:
                related.add(link.target_id)
            elif link.target_id == event_id:
                related.add(link.source_id)
        return tuple(self.events[key] for key in sorted(related)[:limit] if key in self.events)

    def _persist(self, record: dict[str, object]) -> None:
        if not self.path:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeTraceGraph":
        graph = cls(path)
        target = Path(path)
        if not target.exists():
            return graph
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw: dict[str, Any] = json.loads(line)
                if raw.get("record_type") == "link":
                    graph.links.add(
                        TraceCausalLink(
                            source_id=str(raw["source_id"]),
                            target_id=str(raw["target_id"]),
                            relation=str(raw["relation"]),
                        )
                    )
                    continue
                event = TraceEvent(
                    event_id=str(raw["event_id"]),
                    timestamp=str(raw["timestamp"]),
                    kind=str(raw["kind"]),
                    path=raw.get("path"),
                    line=int(raw["line"]) if raw.get("line") is not None else None,
                    symbol=raw.get("symbol"),
                    message=str(raw.get("message", "")),
                    data=tuple(tuple(item) for item in raw.get("data", [])),
                    parent_id=raw.get("parent_id"),
                )
                graph.events[event.event_id] = event
            except (ValueError, TypeError, KeyError):
                continue
        return graph
