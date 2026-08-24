import pytest

from myai import AIEngine, ChatRequest, Document


@pytest.fixture(autouse=True)
def deterministic_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYAI_EMBEDDING_PROVIDER", "deterministic")


def test_engine_returns_valid_v71_response() -> None:
    result = AIEngine().generate(ChatRequest(message="Hello MY-AI"))
    assert result.version == "v7.1"
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
    assert result.version == "v7.1"


def test_engine_uses_semantic_knowledge_context() -> None:
    engine = AIEngine()
    chunks = engine.add_document(
        Document(source="policy.md", text="Refunds are available within 30 days.")
    )
    assert chunks == 1

    retrieved = engine.retrieve("How long are refunds available?", top_k=1)
    assert len(retrieved) == 1
    assert retrieved[0].source == "policy.md"


def test_coding_agent_context_contains_exact_source_slice(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "auth.py"
    source.write_text(
        "class AuthService:\n"
        "    def refresh_token(self):\n"
        "        return 'NEW_TOKEN'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MYAI_CODE_INDEX_ROOT", str(tmp_path))
    monkeypatch.setenv("MYAI_CODE_INDEX_SNAPSHOT_PATH", str(tmp_path / "code-index.json"))

    engine = AIEngine()
    captured: list[str] = []

    class CapturingRuntime:
        def run(self, objective, task_kind, context):
            captured.extend(message.content for message in context)
            return type("Result", (), {"artifact": type("Artifact", (), {"output": "ok"})()})()

    engine.agent_runtime = CapturingRuntime()
    engine.run_agent_task(ChatRequest(message="Fix refresh_token in auth.py"))

    combined = "\n".join(captured)
    assert "refresh_token" in combined
    assert "return 'NEW_TOKEN'" in combined
