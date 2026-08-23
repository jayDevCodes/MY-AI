from myai.config import get_settings


def test_default_settings() -> None:
    settings = get_settings()
    assert settings.app_name == "MY-AI"
    assert settings.environment == "development"
    assert settings.model_provider == "local"
    assert settings.model_name == "placeholder-v1"
    assert "accurate" in settings.system_prompt
