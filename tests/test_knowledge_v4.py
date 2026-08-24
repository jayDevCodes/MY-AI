from pathlib import Path

from myai import Document
from myai.embeddings import DeterministicEmbeddingModel
from myai.knowledge import SQLiteVectorStore, chunk_text


def test_chunk_text_has_overlap() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=6, overlap=2)
    assert chunks == ["abcdef", "efghij"]


def test_sqlite_vector_store_persists_and_retrieves(tmp_path: Path) -> None:
    db_path = tmp_path / "knowledge.sqlite3"
    first = SQLiteVectorStore(db_path, DeterministicEmbeddingModel())
    document = Document(source="notes.txt", text="Python is a programming language.")
    assert first.add(document, chunk_size=200, overlap=0) == 1

    second = SQLiteVectorStore(db_path, DeterministicEmbeddingModel())
    results = second.search("Python programming", top_k=1)
    assert len(results) == 1
    assert results[0].source == "notes.txt"
    assert results[0].score > 0
