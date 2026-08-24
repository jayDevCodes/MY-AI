from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return one normalized embedding vector per document."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return one normalized embedding vector for a query."""


class DeterministicEmbeddingModel:
    """Stable dependency-light embedding fallback used by CI and offline tests."""

    dimension = 64

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        values = np.zeros(DeterministicEmbeddingModel.dimension, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % DeterministicEmbeddingModel.dimension
            values[index] += 1.0
        norm = np.linalg.norm(values)
        return values if norm == 0 else values / norm

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class SentenceTransformerEmbeddingModel:
    """Semantic embedding adapter backed by Sentence Transformers."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if hasattr(self.model, "encode_document"):
            vectors = self.model.encode_document(
                list(texts), normalize_embeddings=True, convert_to_numpy=True
            )
        else:
            vectors = self.model.encode(
                list(texts), normalize_embeddings=True, convert_to_numpy=True
            )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        if hasattr(self.model, "encode_query"):
            vector = self.model.encode_query(text, normalize_embeddings=True, convert_to_numpy=True)
        else:
            vector = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return np.asarray(vector, dtype=np.float32)
