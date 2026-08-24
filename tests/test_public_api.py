from myai.knowledge import InMemoryKnowledgeStore, SQLiteVectorStore, chunk_text


def test_knowledge_api_uses_the_v5_semantic_implementation() -> None:
    """Guard against reintroducing a package that shadows ``knowledge.py``."""
    assert callable(chunk_text)
    assert InMemoryKnowledgeStore.__module__ == "myai.knowledge"
    assert SQLiteVectorStore.__module__ == "myai.knowledge"
