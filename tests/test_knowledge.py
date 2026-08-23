from myai.knowledge import Document, InMemoryKnowledgeStore
from myai.knowledge.chunker import chunk_document


def test_chunk_document_respects_overlap() -> None:
    document = Document(source="demo.txt", text="one two three four five six")
    chunks = chunk_document(document, chunk_size=4, overlap=1)
    assert chunks == ["one two three four", "four five six"]


def test_retrieval_returns_relevant_source() -> None:
    store = InMemoryKnowledgeStore()
    store.add(Document(source="policy.md", text="Refunds are available within 30 days."))
    store.add(Document(source="weather.md", text="Tomorrow will be sunny and warm."))

    results = store.search("refund 30 days", top_k=2)

    assert results
    assert results[0].source == "policy.md"
    assert results[0].score > 0


def test_empty_query_is_safe() -> None:
    store = InMemoryKnowledgeStore()
    store.add(Document(source="demo.txt", text="hello world"))
    assert store.search("", top_k=5) == []
