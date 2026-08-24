from myai.config import get_settings


def test_default_settings(monkeypatch) -> None:
    monkeypatch.delenv("MYAI_EMBEDDING_PROVIDER", raising=False)
    settings = get_settings()
    assert settings.app_name == "MY-AI"
    assert settings.environment == "development"
    assert settings.model_provider == "local"
    assert settings.model_name == "placeholder-v8"
    assert settings.fast_model_name == "placeholder-v8-fast"
    assert settings.balanced_model_name == "placeholder-v8-balanced"
    assert settings.frontier_model_name == "placeholder-v8-frontier"
    assert settings.embedding_provider == "sentence-transformers"
    assert settings.agent_mode == "auto"
    assert settings.agent_max_depth == 3
    assert settings.agent_max_parallel == 4
    assert settings.code_index_enabled is True
    assert settings.code_index_snapshot_path == "data/code_index.json"
    assert settings.repair_memory_path == "data/repair_memory.jsonl"
    assert "causal repository twin" in settings.system_prompt
