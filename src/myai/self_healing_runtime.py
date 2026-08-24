from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class FailureSignature:
    value: str
    error_type: str
    normalized_message: str
    primary_symbol: str = ""


@dataclass(frozen=True)
class RepairEpisode:
    episode_id: str
    signature: FailureSignature
    detected_at: float
    reproduced: bool
    attempts: int
    status: str
    lesson: str = ""


class FailureSignatureStore:
    """Persistent compact index of previously observed failure signatures."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, signature: FailureSignature) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(signature), ensure_ascii=False) + "\n")

    def similar(self, signature: FailureSignature, limit: int = 5) -> tuple[FailureSignature, ...]:
        if not self.path.exists():
            return ()
        rows: list[tuple[float, FailureSignature]] = []
        left = set(signature.normalized_message.split())
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                item = FailureSignature(
                    value=str(raw["value"]),
                    error_type=str(raw["error_type"]),
                    normalized_message=str(raw["normalized_message"]),
                    primary_symbol=str(raw.get("primary_symbol", "")),
                )
            except (ValueError, TypeError, KeyError):
                continue
            right = set(item.normalized_message.split())
            score = len(left & right) / max(1, len(left | right))
            if item.error_type == signature.error_type:
                score += 1.0
            rows.append((score, item))
        rows.sort(key=lambda pair: -pair[0])
        return tuple(item for _, item in rows[:limit])


class SelfHealingRuntime:
    """Bounded self-healing supervisor. It proposes/replays repairs but never auto-promotes code."""

    def __init__(self, *, episode_path: str | Path, signature_store: FailureSignatureStore) -> None:
        self.episode_path = Path(episode_path)
        self.episode_path.parent.mkdir(parents=True, exist_ok=True)
        self.signature_store = signature_store

    def signature(self, error_type: str, message: str, primary_symbol: str = "") -> FailureSignature:
        normalized = " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in message.casefold()).split())
        value = sha256(f"{error_type.casefold()}|{normalized}|{primary_symbol.casefold()}".encode("utf-8")).hexdigest()[:24]
        return FailureSignature(value, error_type, normalized, primary_symbol)

    def record_episode(self, episode: RepairEpisode) -> None:
        with self.episode_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(episode), ensure_ascii=False) + "\n")

    def reproduce(self, command: Sequence[str], *, timeout_seconds: float = 30.0) -> tuple[bool, str]:
        """Replay a failure in a bounded subprocess and capture stdout/stderr."""
        try:
            completed = subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                timeout=max(0.1, timeout_seconds),
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"reproduction-error:{type(exc).__name__}:{exc}"
        output = (completed.stdout + "\n" + completed.stderr).strip()
        return completed.returncode != 0, output

    def verified_repair(
        self,
        *,
        signature: FailureSignature,
        reproduce: Callable[[], tuple[bool, str]],
        validate: Callable[[], bool],
        lesson: str = "",
    ) -> RepairEpisode:
        """Run bounded reproduction + validation. Promotion is intentionally external."""
        self.signature_store.record(signature)
        reproduced, _ = reproduce()
        status = "reproduced" if reproduced else "not-reproduced"
        attempts = 0
        if reproduced:
            attempts = 1
            status = "verified" if validate() else "validation-failed"
        episode = RepairEpisode(
            episode_id=f"{signature.value}-{int(time.time() * 1000)}",
            signature=signature,
            detected_at=time.time(),
            reproduced=reproduced,
            attempts=attempts,
            status=status,
            lesson=lesson,
        )
        self.record_episode(episode)
        return episode
