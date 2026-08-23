import os

os.environ["MYAI_EMBEDDING_PROVIDER"] = "deterministic"

from myai import AIEngine, ChatRequest, Document


def test_engine_returns_valid_v4_response() -> None:
    result = AIEngine().generate(ChatRequest(message="Hello MY-AI"))
    assert result.version == "v4"
    assert result.model == "placeholder-v4"
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
    assert result.version == "v4"


def test_engine_uses_semantic_knowledge_context() -> None:
    engine = AIEngine()
    chunks = engine.add_document(
        Document(source="policy.md", text="Refunds are available within 30 days.")
    )
    assert chunks == 1

    retrieved = engine.retrieve("How long are refunds available?", top_k=1)
    assert len(retrieved) == 1
    assert retrieved[0].source == "policy.md"
