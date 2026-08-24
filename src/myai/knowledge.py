from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .embeddings import DeterministicEmbeddingModel, EmbeddingModel


@dataclass(frozen=True)
class Document:
    source: str
    text: str
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    text: str
    chunk_index: int
    score: float
    metadata: dict[str, str]


class KnowledgeStore(Protocol):
    def add(self, document: Document, *, chunk_size: int, overlap: int) -> int: ...
    def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]: ...


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    # Do not emit a final fragment made only from the overlap of the preceding
    # chunk. It adds duplicate context without contributing new content.
    while start == 0 or start + overlap < len(cleaned):
        chunks.append(cleaned[start : start + chunk_size])
        start += step
    return chunks


class InMemoryKnowledgeStore:
    """Small deterministic store kept for unit tests and offline development."""

    def __init__(self) -> None:
        self._items: list[tuple[str, int, str, dict[str, str]]] = []
        self._embedder = DeterministicEmbeddingModel()

    def add(self, document: Document, *, chunk_size: int = 800, overlap: int = 120) -> int:
        chunks = chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        metadata = document.metadata or {}
        self._items.extend(
            (document.source, index, text, metadata) for index, text in enumerate(chunks)
        )
        return len(chunks)

    def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0 or not query.strip():
            return []
        query_vector = self._embedder.embed_query(query)
        scored: list[RetrievedChunk] = []
        for source, index, text, metadata in self._items:
            vector = self._embedder.embed_query(text)
            score = float(np.dot(query_vector, vector))
            scored.append(RetrievedChunk(source, text, index, score, metadata))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


class SQLiteVectorStore:
    """Persistent cosine-similarity vector store backed by SQLite + NumPy."""

    def __init__(self, path: str | Path, embedder: EmbeddingModel) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        with sqlite3.connect(self.path) as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            db.commit()

    def add(self, document: Document, *, chunk_size: int, overlap: int) -> int:
        chunks = chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0
        embeddings = self.embedder.embed_documents(chunks)
        metadata = json.dumps(document.metadata or {}, sort_keys=True)
        with sqlite3.connect(self.path) as db:
            db.executemany(
                (
                    "INSERT INTO chunks(source, chunk_index, text, metadata, embedding) "
                    "VALUES (?, ?, ?, ?, ?)"
                ),
                [
                    (
                        document.source,
                        index,
                        text,
                        metadata,
                        vector.astype(np.float32).tobytes(),
                    )
                    for index, (text, vector) in enumerate(zip(chunks, embeddings, strict=True))
                ],
            )
        return len(chunks)

    def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0 or not query.strip():
            return []
        query_vector = self.embedder.embed_query(query).astype(np.float32)
        rows: list[tuple[int, str, int, str, str, bytes]]
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                "SELECT id, source, chunk_index, text, metadata, embedding FROM chunks"
            ).fetchall()

        scored: list[RetrievedChunk] = []
        query_norm = float(np.linalg.norm(query_vector))
        for _, source, index, text, metadata_json, raw_embedding in rows:
            vector = np.frombuffer(raw_embedding, dtype=np.float32)
            vector_norm = float(np.linalg.norm(vector))
            score = 0.0 if query_norm == 0 or vector_norm == 0 else float(
                np.dot(query_vector, vector) / (query_norm * vector_norm)
            )
            scored.append(
                RetrievedChunk(
                    source=source,
                    text=text,
                    chunk_index=index,
                    score=score,
                    metadata=json.loads(metadata_json),
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
