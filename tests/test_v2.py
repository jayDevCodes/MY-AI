from myai.engine import AIEngine
from myai.memory import ConversationMemory
from myai.providers import FallbackProvider
from myai.schemas import ChatMessage, ChatRequest


def test_fallback_provider_is_deterministic() -> None:
    provider = FallbackProvider()
    result = provider.generate([ChatMessage(role="user", content="hello")])
    assert "MY-AI V7 fallback is active" in result
    assert "5 characters" in result


def test_memory_is_bounded() -> None:
    memory = ConversationMemory(max_messages=2)
    memory.add(ChatMessage(role="user", content="one"))
    memory.add(ChatMessage(role="assistant", content="two"))
    memory.add(ChatMessage(role="user", content="three"))

    assert [item.content for item in memory.messages()] == ["two", "three"]


def test_engine_returns_v7_response(monkeypatch) -> None:
    monkeypatch.setenv("MYAI_EMBEDDING_PROVIDER", "deterministic")
    engine = AIEngine()
    response = engine.generate(ChatRequest(message="hello"))

    assert response.version == "v7"
    assert response.model == "placeholder-v7"
    assert response.text


def test_engine_builds_system_prompt(monkeypatch) -> None:
    monkeypatch.setenv("MYAI_EMBEDDING_PROVIDER", "deterministic")
    engine = AIEngine()
    messages = engine._build_messages("hello", [])

    assert messages[0].role == "system"
    assert messages[-1].role == "user"
    assert messages[-1].content == "hello"
