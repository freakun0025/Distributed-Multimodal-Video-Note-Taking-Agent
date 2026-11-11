"""Settings from env / .env — every environment variable lives here (ARCHITECTURE §12)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LlmProvider = Literal["gemini", "ollama", "fake"]
# Gemini thinking levels; gemini-3.8-flash rejects "minimal" (ARCHITECTURE §8.2).
ThinkingLevel = Literal["minimal", "low", "medium", "high"]
# Per-image media resolution sent to Gemini (measured 2026-11-08: low 264, medium 527 tokens).
ImageResolution = Literal["low", "medium", "high"]

# Default LLM_CHUNK_CHARS per provider when the variable is unset (ARCHITECTURE §8.2, §12).
_DEFAULT_CHUNK_CHARS: dict[str, int] = {"gemini": 60_000, "ollama": 12_000, "fake": 12_000}


class LlmModelSpec(BaseModel):
    """One link of LLM_CHAIN: the model, its thinking level and its per-request timeout."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1)
    thinking: ThinkingLevel
    timeout_s: float = Field(gt=0)

    @classmethod
    def parse(cls, entry: str) -> LlmModelSpec:
        """Parse model:thinking:timeout_s, split from the right so a model id may contain colons
        (OpenRouter-style ids such as vendor/model:free)."""
        parts = entry.strip().rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError(f"LLM_CHAIN entry {entry!r} is not model:thinking:timeout_s")
        model, thinking, timeout_s = (part.strip() for part in parts)
        return cls.model_validate({"model": model, "thinking": thinking, "timeout_s": timeout_s})

    def __str__(self) -> str:
        return f"{self.model}:{self.thinking}:{self.timeout_s:g}"


# Measured on 2026-11-10 (ARCHITECTURE §8.2, D36): 3.6 Flash wrote the best notes in 19-30 s;
# 3.8 Flash (rejects "minimal") once took ~94 s; Flash-Lite answers in 11-22 s at "medium".
DEFAULT_LLM_CHAIN = (
    LlmModelSpec(model="gemini-3.6-flash", thinking="low", timeout_s=90),
    LlmModelSpec(model="gemini-3.8-flash", thinking="low", timeout_s=120),
    LlmModelSpec(model="gemini-3.5-flash-lite", thinking="medium", timeout_s=60),
)
# Settings replaced by LLM_CHAIN / LLM_STEP_DOWN_FROM_ATTEMPT (D36). Refused if still set, so a
# stale .env cannot silently run a different chain.
REPLACED_SETTINGS = {
    "LLM_MODEL": "LLM_CHAIN",
    "LLM_FALLBACK_MODEL": "LLM_CHAIN",
    "LLM_THINKING_LEVEL": "LLM_CHAIN",
    "LLM_TIMEOUT_S": "LLM_CHAIN",
    "LLM_FALLBACK_THINKING_LEVEL": "LLM_CHAIN",
    "LLM_FALLBACK_TIMEOUT_S": "LLM_CHAIN",
    "LLM_FALLBACK_FROM_ATTEMPT": "LLM_STEP_DOWN_FROM_ATTEMPT",
}


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
    # Ordered models, "model:thinking:timeout_s" comma-separated (D36). Each call goes to the
    # first model no worker has benched; see ARCHITECTURE §8.2 for the measurements behind it.
    # Timeouts are also sent as X-Server-Timeout, so Google itself gives up at that point.
    llm_chain: Annotated[tuple[LlmModelSpec, ...], NoDecode] = DEFAULT_LLM_CHAIN
    # Earlier attempts wait out one 5xx/timeout through the delay queues (the model may recover);
    # from this attempt on, any 5xx/timeout steps down to the next model at once (D31, D36).
    llm_step_down_from_attempt: int = Field(default=3, ge=1)
    # A model that fails this many times with 5xx/timeout within LLM_STRIKE_WINDOW_S (any job,
    # any worker) is benched for LLM_MODEL_COOLDOWN_S: every worker skips it meanwhile.
    llm_model_strikes: int = Field(default=2, ge=1)
    llm_strike_window_s: float = Field(default=600, gt=0)
    llm_model_cooldown_s: float = Field(default=600, gt=0)
    # A model whose daily quota is spent is benched until midnight here (Gemini resets RPD at
    # midnight Pacific). Looked up when the worker starts.
    llm_quota_reset_tz: str = "America/Los_Angeles"
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "qwen3:8b"
    llm_chunk_chars: int | None = Field(default=None, gt=0)
    llm_max_repairs: int = Field(default=1, ge=0)

    # Rate limits — keep below the real provider limits
    llm_rpm: float = Field(default=4, gt=0)
    llm_tpm: int = Field(default=200_000, gt=0)
    notion_rps: float = Field(default=2.5, gt=0)
    youtube_rpm: float = Field(default=30, gt=0)
    # Bucket capacity = allowed burst. LLM 1 + LLM_RPM 4 admits at most 5 calls in any minute.
    llm_burst: float = Field(default=1, ge=1)
    notion_burst: float = Field(default=3, ge=1)
    youtube_burst: float = Field(default=2, ge=1)
    # Per-request socket timeout for yt-dlp and youtube-transcript-api (neither has one by default).
    youtube_timeout_s: float = Field(default=30, gt=0)
    limiter_max_wait_s: float = Field(default=20, ge=0)
    redis_timeout_s: float = Field(default=2, gt=0)

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
    notion_api_url: str = "https://api.notion.com"  # FAKE_APIS=true uses {FAKES_URL}/notion
    notion_timeout_s: float = Field(default=30, gt=0)  # per request; uploads are small JPEGs

    # Security
    token_encryption_key: SecretStr | None = None

    # Pipeline
    max_video_seconds: int = Field(default=1200, gt=0)
    # Frames per job (ARCHITECTURE §8.2: 3-8). The LLM must pick at least one frame per
    # MIN_FRAME_EVERY_S of video, never fewer than MIN_FRAMES nor more than MAX_FRAMES (CP7c).
    min_frames: int = Field(default=3, ge=0)
    max_frames: int = Field(default=8, ge=0)
    min_frame_every_s: int = Field(default=180, gt=0)
    # Study-notes depth (CP7c, D35, D37): each part must have at least one section per
    # NOTES_SECTION_EVERY_S and NOTES_BULLETS_PER_MINUTE bullets per minute, spread over its
    # sections. Gemini enforces these as minItems and every model measured writes about the
    # floor, so the floor sets the depth (the prompt asks for more where the lecture has more).
    # Each counted item enlarges the schema Gemini compiles, so the floors are capped.
    notes_section_every_s: int = Field(default=150, gt=0)
    notes_bullets_per_minute: float = Field(default=2.5, ge=0)
    notes_floor_max_sections: int = Field(default=8, ge=1)
    notes_floor_max_bullets: int = Field(default=8, ge=1)
    # Self-test questions (D38): NOTES_QUESTIONS_MIN..MAX per video, spread over its parts
    # (schema minItems/maxItems per part, the merged list capped at the max again).
    notes_questions_min: int = Field(default=5, ge=0)
    notes_questions_max: int = Field(default=8, ge=1)
    # Frames closer together than this show the same slide; the later one is dropped.
    frame_min_gap_s: int = Field(default=10, ge=0)
    # EXTRACT_FRAMES (ARCHITECTURE §8.3, D33): one download at this height, then per pick
    # FRAME_BURST grabs inside the same second, keeping the least blurry.
    frames_video_height: int = Field(default=720, gt=0)
    frame_burst: int = Field(default=3, ge=1)
    frame_jpeg_quality: int = Field(default=3, ge=2, le=31)  # ffmpeg -q:v, lower is better
    # Blank frame: mean luma below this (video black is 16 on the 16-235 scale), or almost no
    # spread between darkest and brightest pixel (a fade to one colour).
    frame_blank_luma: float = Field(default=24, ge=0)
    frame_flat_range: float = Field(default=24, ge=0)
    ffmpeg_timeout_s: float = Field(default=60, gt=0)
    # Frame-aware notes (CP7b, D34): SUMMARIZE shows the LLM up to CANDIDATE_MAX real frames from
    # a small download — the strongest scene cut per time slot (taken CANDIDATE_SETTLE_S after the
    # cut, once a transition has settled), else the slot's middle — and picks only among them.
    candidates_video_height: int = Field(default=480, gt=0)
    candidate_max: int = Field(default=36, ge=0)  # 0 = captions only
    candidate_min_gap_s: int = Field(default=15, gt=0)
    candidate_settle_s: float = Field(default=1.5, ge=0)
    candidate_jpeg_quality: int = Field(default=5, ge=2, le=31)
    candidate_snap_s: int = Field(default=2, ge=0)  # a pick this close to a candidate is it
    scene_threshold: float = Field(default=10, ge=0)  # ffmpeg scdet score for a cut (its default)
    scene_detect_timeout_s: float = Field(default=180, gt=0)
    llm_image_resolution: ImageResolution = "medium"
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
    # Where the fakes service listens (profile `fakes`); used only when FAKE_APIS=true.
    fakes_url: str = "http://fakes:8090"
    log_level: str = "INFO"

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

    @model_validator(mode="before")
    @classmethod
    def _refuse_replaced_settings(cls, data: Any) -> Any:
        given = {str(key).upper() for key in data} if isinstance(data, dict) else set()
        given |= {key.upper() for key, value in os.environ.items() if value}
        if stale := sorted(given & REPLACED_SETTINGS.keys()):
            replacements = ", ".join(f"{old} -> {REPLACED_SETTINGS[old]}" for old in stale)
            raise ValueError(f"settings replaced in D36, remove them from .env: {replacements}")
        return data

    @field_validator("llm_chain", mode="before")
    @classmethod
    def _parse_chain(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(LlmModelSpec.parse(entry) for entry in _split_csv(value))
        return value

    @field_validator("llm_chain")
    @classmethod
    def _check_chain(cls, value: tuple[LlmModelSpec, ...]) -> tuple[LlmModelSpec, ...]:
        models = [spec.model for spec in value]
        if not models:
            raise ValueError("LLM_CHAIN needs at least one model:thinking:timeout_s")
        if len(set(models)) != len(models):
            raise ValueError(f"LLM_CHAIN lists a model twice: {models}")
        return value

    @model_validator(mode="after")
    def _check_questions(self) -> Settings:
        if self.notes_questions_min > self.notes_questions_max:
            raise ValueError("NOTES_QUESTIONS_MIN must not exceed NOTES_QUESTIONS_MAX")
        return self

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
