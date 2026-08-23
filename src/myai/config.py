from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    app_name: str = "MY-AI"
    environment: str = "development"
    log_level: str = "INFO"
    model_provider: str = "local"
    model_name: str = "placeholder-v3"
    model_base_url: str = "http://localhost:11434"
    model_api_key: str = ""
    model_timeout_seconds: float = 60.0
    memory_max_messages: int = 20
    knowledge_top_k: int = 5
    knowledge_chunk_size: int = 800
    knowledge_chunk_overlap: int = 120
    system_prompt: str = (
        "You are MY-AI V3. Be accurate, explicit about uncertainty, "
        "use retrieved knowledge when supplied, and never invent sources."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MYAI_",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()
