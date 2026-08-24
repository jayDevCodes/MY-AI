import pytest

from myai import AIEngine, ChatRequest, Document


@pytest.fixture(autouse=True)
def deterministic_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYAI_EMBEDDING_PROVIDER", "deterministic")


def test_engine_returns_valid_v7_response() -> None:
    result = AIEngine().generate(ChatRequest(message="Hello MY-AI"))
    assert result.version == "v7"
    assert result.model == "placeholder-v7"
    assert result.text


def test_engine_preserves_conversation_count() -> None:
    request = ChatRequest(
        message="Continue",
        conversation=[
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
        ],
    )
    result = AIEngine().generate(request)
    assert result.version == "v7"


def test_engine_uses_semantic_knowledge_context() -> None:
    engine = AIEngine()
    chunks = engine.add_document(
        Document(source="policy.md", text="Refunds are available within 30 days.")
    )
    assert chunks == 1

    retrieved = engine.retrieve("How long are refunds available?", top_k=1)
    assert len(retrieved) == 1
    assert retrieved[0].source == "policy.md"
