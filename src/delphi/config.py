"""Environment-backed configuration with zero-cost defaults."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DELPHI_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://delphi:delphi@localhost:5436/delphi"
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    http_timeout_seconds: float = 30.0
    random_seed: int = 20260802

    # Explicit, never inferred from data freshness. This prevents a baked demo from being
    # presented as live merely because its generated_at timestamp is recent.
    snapshot_mode: Literal["live", "demo", "replay"] = "replay"

    # Language generation is outside the decision path. The default is deterministic and
    # requires no model, hosted service, API key, or network request.
    llm_backend: Literal["template", "ollama"] = "template"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"


@lru_cache
def get_settings() -> Settings:
    return Settings()
