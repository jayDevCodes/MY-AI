import re
from collections import Counter

from .chunker import chunk_document
from .models import Document, RetrievedChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


class InMemoryKnowledgeStore:
    """Small dependency-free lexical retrieval store for V3."""

    def __init__(self) -> None:
        self._chunks: list[tuple[str, int, str]] = []

    def add(self, document: Document, *, chunk_size: int = 800, overlap: int = 120) -> int:
        chunks = chunk_document(document, chunk_size=chunk_size, overlap=overlap)
        for index, chunk in enumerate(chunks):
            self._chunks.append((document.source, index, chunk))
        return len(chunks)

    def clear(self) -> None:
        self._chunks.clear()

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        if not query.strip() or top_k <= 0:
            return []

        query_tokens = Counter(_tokens(query))
        if not query_tokens:
            return []

        scored: list[RetrievedChunk] = []
        for source, index, text in self._chunks:
            chunk_tokens = Counter(_tokens(text))
            overlap = sum(min(query_tokens[token], chunk_tokens[token]) for token in query_tokens)
            if overlap == 0:
                continue
            score = overlap / max(1, sum(query_tokens.values()))
            scored.append(
                RetrievedChunk(source=source, text=text, score=score, chunk_index=index)
            )

        scored.sort(key=lambda item: (-item.score, item.source, item.chunk_index))
        return scored[:top_k]
