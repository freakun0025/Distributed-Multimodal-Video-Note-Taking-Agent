"""Pydantic contracts for gateway, worker, stages and extension (ARCHITECTURE §5, §6.2, §8, §10).

REST shapes follow Spec §5 exactly; `waiting` on GET /jobs/{id} is our one additive field.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from common.errors import ErrorCode, WarningCode
from common.state import JobStatus, Stage

# --------------------------------------------------------------------------------------------
# Broker message (ARCHITECTURE §6.2)
# --------------------------------------------------------------------------------------------


class Trace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    first_enqueued_at: datetime


class StageMessage(BaseModel):
    """Body of every message on jobs.work / jobs.retry.* / jobs.dlq."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: UUID
    stage: Stage
    attempt: int = Field(ge=1)  # attempts of *this* stage; resets to 1 on advance
    notion_token_enc: str = Field(min_length=1, repr=False)  # Fernet ciphertext, never plaintext
    notion_parent_id: str | None = None
    trace: Trace | None = None

    def to_body(self) -> bytes:
        return self.model_dump_json().encode()

    @classmethod
    def from_body(cls, body: bytes) -> StageMessage:
        return cls.model_validate_json(body)

    def for_stage(self, stage: Stage) -> StageMessage:
        """Message for the next stage: same job, attempt reset to 1."""
        return self.model_copy(update={"stage": stage, "attempt": 1})

    def for_retry(self) -> StageMessage:
        return self.model_copy(update={"attempt": self.attempt + 1})


# --------------------------------------------------------------------------------------------
# REST API (Spec §5, ARCHITECTURE §10)
# --------------------------------------------------------------------------------------------


class ErrorInfo(BaseModel):
    code: ErrorCode
    message: str


class ErrorBody(BaseModel):
    """`{"error": {"code", "message"}}` — the body of every 4xx/5xx response."""

    error: ErrorInfo


class JobWarning(BaseModel):
    code: WarningCode
    timestamp_sec: int | None = None
    message: str | None = None

    @model_serializer(mode="wrap")
    def _drop_nulls(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Spec example: {"code": "FRAME_EXTRACTION_FAILED", "timestamp_sec": 461} — no null keys.
        data: dict[str, Any] = handler(self)
        return {key: value for key, value in data.items() if value is not None}


class JobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    youtube_url: str = Field(max_length=2048)
    # Optional in the schema so a missing token yields 400 MISSING_NOTION_TOKEN, not a 422.
    notion_token: SecretStr | None = None
    notion_parent_id: str | None = Field(default=None, max_length=200)


class JobCreatedResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    video_id: str


class JobResult(BaseModel):
    notion_page_url: str
    frames_requested: int = Field(ge=0)
    frames_published: int = Field(ge=0)


class WaitingInfo(BaseModel):
    """Set while a retry is scheduled: the popup shows "waiting", not "failed"."""

    until: datetime
    reason: str  # e.g. RATE_LIMITED:llm


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    video_title: str | None = None
    duration_sec: int | None = None
    warnings: list[JobWarning] = Field(default_factory=list)
    error: ErrorInfo | None = None
    result: JobResult | None = None
    waiting: WaitingInfo | None = None


class JobResultsResponse(BaseModel):
    status: JobStatus
    result: JobResult | None = None
    warnings: list[JobWarning] = Field(default_factory=list)
    error: ErrorInfo | None = None


ComponentHealth = Literal["ok", "down"]


class HealthResponse(BaseModel):
    gateway: ComponentHealth = "ok"
    postgres: ComponentHealth
    rabbitmq: ComponentHealth
    redis: ComponentHealth


class NotionExchangeRequest(BaseModel):
    code: str = Field(min_length=1, repr=False)


class NotionRefreshRequest(BaseModel):
    refresh_token: SecretStr


class NotionTokenResponse(BaseModel):
    """Passed straight back to the extension; the gateway never stores it."""

    access_token: str = Field(repr=False)
    refresh_token: str | None = Field(default=None, repr=False)
    workspace_name: str | None = None
    bot_id: str | None = None
    expires_in: int | None = None


# --------------------------------------------------------------------------------------------
# Pipeline artefacts (ARCHITECTURE §5 artefacts, §8)
# --------------------------------------------------------------------------------------------


class VideoMeta(BaseModel):
    """meta.json"""

    video_id: str
    title: str
    duration_sec: int = Field(ge=0)
    channel: str | None = None
    language: str | None = None


class TranscriptSegment(BaseModel):
    start: float = Field(ge=0)
    duration: float = Field(ge=0)
    text: str


class Transcript(BaseModel):
    """transcript.json"""

    source: Literal["captions", "stt"]
    language: str | None = None
    segments: list[TranscriptSegment]


class FramePick(BaseModel):
    t_sec: int = Field(ge=0)
    caption: str = Field(min_length=1)  # what is shown (C-B: every frame has a caption)
    reason: str = Field(min_length=1)  # why it was flagged


class Section(BaseModel):
    heading: str = Field(min_length=1)
    bullets: list[str]


class Synthesis(BaseModel):
    """synthesis.json — also the structured-output schema given to the LLM."""

    title: str
    summary: str
    sections: list[Section]
    frames: list[FramePick]


class FrameResult(BaseModel):
    """One entry of frames/manifest.json: exactly one of `path` / `error` is set."""

    t_sec: int = Field(ge=0)
    path: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _one_outcome(self) -> FrameResult:
        if (self.path is None) == (self.error is None):
            raise ValueError("exactly one of path or error must be set")
        return self

    @property
    def ok(self) -> bool:
        return self.path is not None
