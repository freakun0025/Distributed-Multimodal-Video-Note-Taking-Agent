"""Settings from env / .env — every environment variable lives here (ARCHITECTURE §12)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LlmProvider = Literal["gemini", "ollama", "fake"]
# Gemini thinking levels; gemini-3.8-flash rejects "minimal" (ARCHITECTURE §8.2).
ThinkingLevel = Literal["minimal", "low", "medium", "high"]
# Per-image media resolution sent to Gemini (measured 2026-11-08: low 264, medium 527 tokens).
ImageResolution = Literal["low", "medium", "high"]

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
    llm_model: str = "gemini-3.8-flash"
    llm_fallback_model: str = "gemini-3.5-flash-lite"
    # Earlier attempts wait out a busy primary model (5xx/timeout) through the delay queues;
    # from this attempt on, an overloaded primary hands over to LLM_FALLBACK_MODEL (D31).
    llm_fallback_from_attempt: int = Field(default=3, ge=1)
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "qwen3:8b"
    llm_chunk_chars: int | None = Field(default=None, gt=0)
    llm_max_repairs: int = Field(default=1, ge=0)
    llm_thinking_level: ThinkingLevel = "low"
    # Seen 2026-11-07: gemini-3.8-flash took ~94 s per call (even for a one-word answer) under
    # high demand; a slow success must not be cut off and retried. The SDK also sends this as
    # X-Server-Timeout, so Google itself gives up at this point (a 504 after exactly 240 s).
    llm_timeout_s: float = Field(default=240, gt=0)
    # The fallback model gets its own settings (CP7c): gemini-3.5-flash-lite wrote better notes
    # at "medium" than at "low" in the same ~11-14 s (18-min lecture, 36 images), and a call
    # that answers in ~14 s should not hold a worker for 240 s when it hangs.
    llm_fallback_thinking_level: ThinkingLevel = "medium"
    llm_fallback_timeout_s: float = Field(default=90, gt=0)

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

    # Security
    token_encryption_key: SecretStr | None = None

    # Pipeline
    max_video_seconds: int = Field(default=1200, gt=0)
    # Frames per job (ARCHITECTURE §8.2: 3-8). The LLM must pick at least one frame per
    # MIN_FRAME_EVERY_S of video, never fewer than MIN_FRAMES nor more than MAX_FRAMES (CP7c).
    min_frames: int = Field(default=3, ge=0)
    max_frames: int = Field(default=8, ge=0)
    min_frame_every_s: int = Field(default=180, gt=0)
    # Revision-depth notes (CP7c, D35): each part must have at least one section per
    # NOTES_SECTION_EVERY_S and NOTES_BULLETS_PER_MINUTE bullets per minute, spread over its
    # sections. Gemini enforces these as minItems; each counted item enlarges the schema it
    # compiles, so the floors are capped (the prompt asks for more where the lecture has more).
    notes_section_every_s: int = Field(default=180, gt=0)
    notes_bullets_per_minute: float = Field(default=1.5, ge=0)
    notes_floor_max_sections: int = Field(default=8, ge=1)
    notes_floor_max_bullets: int = Field(default=6, ge=1)
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
