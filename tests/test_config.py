from myai.config import get_settings


def test_default_settings(monkeypatch) -> None:
    monkeypatch.delenv("MYAI_EMBEDDING_PROVIDER", raising=False)
    settings = get_settings()
    assert settings.app_name == "MY-AI"
    assert settings.environment == "development"
    assert settings.model_provider == "local"
    assert settings.model_name == "placeholder-v7"
    assert settings.fast_model_name == "placeholder-v7-fast"
    assert settings.balanced_model_name == "placeholder-v7-balanced"
    assert settings.frontier_model_name == "placeholder-v7-frontier"
    assert settings.embedding_provider == "sentence-transformers"
    assert settings.agent_mode == "auto"
    assert settings.agent_max_depth == 3
    assert settings.agent_max_parallel == 4
    assert settings.code_index_enabled is True
    assert settings.code_index_snapshot_path == "data/code_index.json"
    assert "accurate" in settings.system_prompt
