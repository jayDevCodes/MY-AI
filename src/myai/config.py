from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    app_name: str = "MY-AI"
    environment: str = "development"
    log_level: str = "INFO"
    model_provider: str = "local"
    model_name: str = "placeholder-v9"
    model_base_url: str = "http://localhost:11434"
    model_api_key: str = ""
    model_timeout_seconds: float = 60.0
    fast_model_provider: str = "local"
    fast_model_name: str = "placeholder-v9-fast"
    fast_model_base_url: str = "http://localhost:11434"
    fast_model_api_key: str = ""
    balanced_model_provider: str = "local"
    balanced_model_name: str = "placeholder-v9-balanced"
    balanced_model_base_url: str = "http://localhost:11434"
    balanced_model_api_key: str = ""
    frontier_model_provider: str = "local"
    frontier_model_name: str = "placeholder-v9-frontier"
    frontier_model_base_url: str = "http://localhost:11434"
    frontier_model_api_key: str = ""
    memory_max_messages: int = 20
    memory_store_path: str = "data/cognitive_memory.sqlite3"
    memory_load_limit: int = 100
    knowledge_top_k: int = 5
    knowledge_chunk_size: int = 800
    knowledge_chunk_overlap: int = 120
    knowledge_db_path: str = "data/knowledge.sqlite3"
    embedding_provider: str = "sentence-transformers"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str | None = None
    cognitive_verification: bool = True
    cognitive_max_retries: int = 1
    agent_max_depth: int = 3
    agent_max_nodes: int = 32
    agent_max_parallel: int = 4
    agent_max_retries: int = 1
    agent_mode: str = "auto"
    code_index_enabled: bool = True
    code_index_root: str = "."
    code_index_snapshot_path: str = "data/code_index.json"
    code_context_limit: int = 8
    repair_memory_path: str = "data/repair_memory.jsonl"
    runtime_trace_path: str = "data/runtime_trace.jsonl"
    evolution_memory_path: str = "data/evolution_memory.jsonl"
    capability_ledger_path: str = "data/capability_ledger.json"
    evolution_min_promotion_delta: float = 0.05
    system_prompt: str = (
        "You are MY-AI V9.1. Operate as an evidence-first cognitive mesh. "
        "Use the unified program graph, runtime traces, repository memory, and structured specialist artifacts. "
        "Never invent sources, tool usage, verification, or repository facts. "
        "Prefer the smallest sufficient context and preserve already-verified work. "
        "When strategies disagree, surface uncertainty and request stronger evidence rather than averaging guesses."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MYAI_",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()
