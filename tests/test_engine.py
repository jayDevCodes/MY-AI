from myai import AIEngine, ChatRequest


def test_engine_returns_valid_response() -> None:
    result = AIEngine().generate(ChatRequest(message="Hello MY-AI"))
    assert result.version == "v1"
    assert result.model == "placeholder-v1"
    assert "MY-AI V1" in result.text


def test_engine_preserves_conversation_count() -> None:
    request = ChatRequest(
        message="Continue",
        conversation=[
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
        ],
    )
    result = AIEngine().generate(request)
    assert "Context messages: 2" in result.text
