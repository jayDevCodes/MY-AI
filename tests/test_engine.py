from myai import AIEngine, ChatRequest, Document


def test_engine_returns_valid_v3_response() -> None:
    result = AIEngine().generate(ChatRequest(message="Hello MY-AI"))
    assert result.version == "v3"
    assert result.model == "placeholder-v3"
    assert "MY-AI V3" in result.text


def test_engine_preserves_conversation_count() -> None:
    request = ChatRequest(
        message="Continue",
        conversation=[
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
        ],
    )
    result = AIEngine().generate(request)
    assert result.version == "v3"


def test_engine_uses_knowledge_context() -> None:
    engine = AIEngine()
    chunks = engine.add_document(
        Document(source="policy.md", text="Refunds are available within 30 days.")
    )
    assert chunks == 1

    result = engine.generate(ChatRequest(message="What is the refund period?"))
    assert "Knowledge context blocks: 1" in result.text
