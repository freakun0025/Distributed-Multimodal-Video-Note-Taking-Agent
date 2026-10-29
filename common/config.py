"""Settings from env / .env — every environment variable lives here (ARCHITECTURE §12)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LlmProvider = Literal["gemini", "ollama", "fake"]

# Default LLM_CHUNK_CHARS per provider when the variable is unset (ARCHITECTURE §8.2, §12).
_DEFAULT_CHUNK_CHARS: dict[str, int] = {"gemini": 60_000, "ollama": 12_000, "fake": 12_000}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `KEY=` in .env means "unset", not ""
        extra="ignore",
        frozen=True,
    )

    # Endpoints
    gateway_base_url: str = "http://localhost:8000"
    database_url: str = "postgresql+asyncpg://vna:vna@postgres:5432/vna"
    rabbitmq_url: str = "amqp://vna:vna@rabbitmq:5672/"
    redis_url: str = "redis://redis:6379/0"

    # LLM
    llm_provider: LlmProvider = "gemini"
    llm_api_key: SecretStr | None = None
    llm_model: str = "gemini-3.6-flash"
    llm_fallback_model: str = "gemini-3.5-flash-lite"
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "qwen3:8b"
    llm_chunk_chars: int | None = Field(default=None, gt=0)
    llm_max_repairs: int = Field(default=1, ge=0)

    # Rate limits — keep below the real provider limits
    llm_rpm: float = Field(default=4, gt=0)
    llm_tpm: int = Field(default=200_000, gt=0)
    notion_rps: float = Field(default=2.5, gt=0)
    youtube_rpm: float = Field(default=30, gt=0)
    limiter_max_wait_s: float = Field(default=20, ge=0)

    # Retries / DLQ / outbox
    retry_max_attempts: int = Field(default=5, ge=1)
    retry_delays: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [10, 30, 90])
    outbox_timeout_s: int = Field(default=300, gt=0)
    outbox_sweep_interval_s: float = Field(default=30, gt=0)
    broker_publish_timeout_s: float = Field(default=5, gt=0)
    healthcheck_timeout_s: float = Field(default=2, gt=0)

    # Notion
    notion_client_id: str | None = None
    notion_client_secret: SecretStr | None = None
    notion_redirect_uri: str | None = None
    notion_default_parent_id: str | None = None
    notion_api_version: str = "2026-03-11"

    # Security
    token_encryption_key: SecretStr | None = None

    # Pipeline
    max_video_seconds: int = Field(default=1200, gt=0)
    max_frames: int = Field(default=8, ge=0)
    transcript_langs: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["en", "en-US", "en-GB", "hi"]
    )
    notes_language: str = "English"

    # Speech-to-text
    stt_enabled: bool = False
    stt_url: str = "http://stt:9000"
    whisper_model: str = "small"
    whisper_device: str = "cuda"

    # Runtime
    data_dir: Path = Path("/data")
    fake_apis: bool = False
    log_level: str = "INFO"
    # M1 stub stages sleep this long; removed once every stage is real (M2).
    stub_stage_seconds: float = Field(default=3, ge=0)

    @field_validator("retry_delays", mode="before")
    @classmethod
    def _parse_delays(cls, value: object) -> object:
        if isinstance(value, str):
            return [int(part) for part in _split_csv(value)]
        return value

    @field_validator("retry_delays")
    @classmethod
    def _check_delays(cls, value: list[int]) -> list[int]:
        if not value or any(delay <= 0 for delay in value):
            raise ValueError("RETRY_DELAYS needs at least one positive number of seconds")
        return value

    @field_validator("transcript_langs", mode="before")
    @classmethod
    def _parse_langs(cls, value: object) -> object:
        if isinstance(value, str):
            return _split_csv(value)
        return value

    @property
    def effective_llm_chunk_chars(self) -> int:
        """LLM_CHUNK_CHARS if set, else the provider default."""
        return self.llm_chunk_chars or _DEFAULT_CHUNK_CHARS[self.llm_provider]

    def retry_delay_for(self, attempt: int) -> int:
        """Delay before retry number `attempt` (1-based); attempts past the list reuse the last."""
        index = min(max(attempt, 1), len(self.retry_delays)) - 1
        return self.retry_delays[index]

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
