from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .cognitive_state import Belief, CognitiveState, MemoryItem, MemoryKind


class CognitiveMemoryStore:
    """Durable SQLite store for beliefs and long-lived cognitive memories."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    provenance TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(content, kind)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS beliefs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    statement TEXT NOT NULL UNIQUE,
                    confidence REAL NOT NULL,
                    provenance TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_kind_score ON memories(kind, importance, confidence)"
            )

    @staticmethod
    def _memory_params(memory: MemoryItem) -> tuple[object, ...]:
        return (
            memory.content,
            memory.kind.value,
            memory.importance,
            memory.confidence,
            json.dumps(memory.provenance),
            json.dumps(memory.tags),
            memory.created_at,
        )

    @staticmethod
    def _belief_params(belief: Belief) -> tuple[object, ...]:
        return (
            belief.statement,
            belief.confidence,
            json.dumps(belief.provenance),
            belief.updated_at,
        )

    def remember(self, memory: MemoryItem) -> None:
        with self._connect() as connection:
            self._remember_memory(connection, memory)

    def _remember_memory(self, connection: sqlite3.Connection, memory: MemoryItem) -> None:
        connection.execute(
            """
            INSERT INTO memories(content, kind, importance, confidence, provenance, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(content, kind) DO UPDATE SET
                importance=max(memories.importance, excluded.importance),
                confidence=max(memories.confidence, excluded.confidence),
                provenance=excluded.provenance,
                tags=excluded.tags
            """,
            self._memory_params(memory),
        )

    def remember_belief(self, belief: Belief) -> None:
        with self._connect() as connection:
            self._remember_belief(connection, belief)

    def _remember_belief(self, connection: sqlite3.Connection, belief: Belief) -> None:
        connection.execute(
            """
            INSERT INTO beliefs(statement, confidence, provenance, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(statement) DO UPDATE SET
                confidence=excluded.confidence,
                provenance=excluded.provenance,
                updated_at=excluded.updated_at
            """,
            self._belief_params(belief),
        )

    def load_memories(self, limit: int = 100) -> tuple[MemoryItem, ...]:
        if limit <= 0:
            return ()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT content, kind, importance, confidence, provenance, tags, created_at
                FROM memories
                ORDER BY importance * confidence DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(
            MemoryItem(
                content=row[0],
                kind=MemoryKind(row[1]),
                importance=float(row[2]),
                confidence=float(row[3]),
                provenance=tuple(json.loads(row[4])),
                tags=tuple(json.loads(row[5])),
                created_at=row[6],
            )
            for row in rows
        )

    def load_beliefs(self, limit: int = 100) -> tuple[Belief, ...]:
        if limit <= 0:
            return ()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT statement, confidence, provenance, updated_at
                FROM beliefs
                ORDER BY confidence DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(
            Belief(
                statement=row[0],
                confidence=float(row[1]),
                provenance=tuple(json.loads(row[2])),
                updated_at=row[3],
            )
            for row in rows
        )

    def hydrate(self, state: CognitiveState, limit: int = 100) -> CognitiveState:
        state.memories.extend(self.load_memories(limit))
        state.beliefs.extend(self.load_beliefs(limit))
        return state

    def persist_state(self, state: CognitiveState) -> None:
        """Persist all state changes in one SQLite transaction for better throughput."""
        with self._connect() as connection:
            for memory in state.memories:
                self._remember_memory(connection, memory)
            for belief in state.beliefs:
                self._remember_belief(connection, belief)
