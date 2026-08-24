from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .repository_twin import CausalRepositoryTwin, ImpactSlice


_TRACE_RE = re.compile(r'File ["\'](?P<path>.+?)["\'], line (?P<line>\d+), in (?P<symbol>[^\n]+)')
_ERROR_RE = re.compile(r'^(?P<type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning)):\s*(?P<message>.*)$', re.MULTILINE)


@dataclass(frozen=True)
class FailureFrame:
    path: str
    line: int
    symbol: str


@dataclass(frozen=True)
class FailureEvent:
    error_type: str
    message: str
    frames: tuple[FailureFrame, ...]
    timestamp: str
    source: str = "runtime"


@dataclass(frozen=True)
class CausalDiagnosis:
    error_type: str
    message: str
    primary_frame: FailureFrame | None
    affected_files: tuple[str, ...]
    impact: ImpactSlice
    evidence: tuple[str, ...]
    root_cause_hypothesis: str
    confidence: float


@dataclass(frozen=True)
class RepairMemoryRecord:
    error_type: str
    signature: str
    root_cause: str
    patch_summary: str
    validation: str
    success: bool
    timestamp: str
    evidence: tuple[str, ...] = ()


class RepairMemory:
    """Small persistent experience store: failures, fixes and validation evidence."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: RepairMemoryRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def similar(self, error_type: str, message: str, limit: int = 5) -> tuple[RepairMemoryRecord, ...]:
        if not self.path.exists():
            return ()
        wanted = _signature(error_type, message)
        records: list[RepairMemoryRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                record = RepairMemoryRecord(
                    error_type=str(item["error_type"]),
                    signature=str(item["signature"]),
                    root_cause=str(item["root_cause"]),
                    patch_summary=str(item["patch_summary"]),
                    validation=str(item["validation"]),
                    success=bool(item["success"]),
                    timestamp=str(item["timestamp"]),
                    evidence=tuple(str(value) for value in item.get("evidence", [])),
                )
            except (ValueError, TypeError, KeyError):
                continue
            score = _similarity(wanted, record.signature)
            if record.error_type == error_type:
                score += 1
            records.append((score, record))
        records.sort(key=lambda item: (-item[0], item[1].timestamp), reverse=False)
        return tuple(record for _, record in records[:limit])


class CausalErrorEngine:
    """Turns runtime failures into a small, evidence-backed repair context."""

    def __init__(self, twin: CausalRepositoryTwin, memory: RepairMemory) -> None:
        self.twin = twin
        self.memory = memory

    def parse_failure(self, traceback_text: str) -> FailureEvent:
        frames = tuple(
            FailureFrame(match.group("path"), int(match.group("line")), match.group("symbol").strip())
            for match in _TRACE_RE.finditer(traceback_text)
        )
        error_match = list(_ERROR_RE.finditer(traceback_text))[-1] if _ERROR_RE.search(traceback_text) else None
        error_type = error_match.group("type") if error_match else "RuntimeError"
        message = error_match.group("message").strip() if error_match else traceback_text.strip().splitlines()[-1]
        return FailureEvent(
            error_type=error_type,
            message=message,
            frames=frames,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def diagnose(self, traceback_text: str, *, limit: int = 8) -> CausalDiagnosis:
        event = self.parse_failure(traceback_text)
        primary = event.frames[-1] if event.frames else None
        query = primary.symbol if primary else event.message
        impact = self.twin.impact_slice(query, limit=limit)
        affected = self.twin.affected_files(primary.path) if primary else ()
        history = self.memory.similar(event.error_type, event.message, limit=3)

        evidence = [f"runtime:{event.error_type}: {event.message}"]
        if primary:
            evidence.append(f"frame:{primary.path}:{primary.line}:{primary.symbol}")
        evidence.extend(f"affected:{path}" for path in affected[:limit])
        evidence.extend(f"history:{record.root_cause}" for record in history if record.success)

        hypothesis = (
            history[0].root_cause
            if history and history[0].success
            else f"Investigate {query} at the primary runtime frame and its dependency impact slice."
        )
        confidence = min(0.99, 0.45 + (0.2 if primary else 0) + (0.15 if impact.nodes else 0) + (0.15 if history else 0))
        return CausalDiagnosis(
            error_type=event.error_type,
            message=event.message,
            primary_frame=primary,
            affected_files=affected,
            impact=impact,
            evidence=tuple(evidence),
            root_cause_hypothesis=hypothesis,
            confidence=confidence,
        )


def _signature(error_type: str, message: str) -> str:
    normalized = re.sub(r"\d+", "#", f"{error_type}:{message}".casefold())
    return " ".join(normalized.split())


def _similarity(left: str, right: str) -> float:
    a = set(left.split())
    b = set(right.split())
    return len(a & b) / max(1, len(a | b))
