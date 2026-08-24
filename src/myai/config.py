from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    app_name: str = "MY-AI"
    environment: str = "development"
    log_level: str = "INFO"
    model_provider: str = "local"
    model_name: str = "placeholder-v5"
    model_base_url: str = "http://localhost:11434"
    model_api_key: str = ""
    model_timeout_seconds: float = 60.0
    memory_max_messages: int = 20
    knowledge_top_k: int = 5
    knowledge_chunk_size: int = 800
    knowledge_chunk_overlap: int = 120
    knowledge_db_path: str = "data/knowledge.sqlite3"
    embedding_provider: str = "sentence-transformers"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str | None = None
    system_prompt: str = (
        "You are MY-AI V5. Be accurate, explicit about uncertainty, "
        "use retrieved knowledge when supplied, cite its source metadata when relevant, "
        "and never invent sources or claim tools were used when they were not."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MYAI_",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()
