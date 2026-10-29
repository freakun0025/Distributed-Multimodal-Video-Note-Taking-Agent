"""StageContext and StageOutcome passed to every stage (ARCHITECTURE §8)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from structlog.typing import FilteringBoundLogger

from common.config import Settings
from common.models import JobWarning, StageMessage
from common.state import JobStatus

# Job columns a stage may set through StageOutcome.job_updates. Never `status`: only
# common.state.transition() writes it (invariant 5).
JOB_UPDATE_COLUMNS: frozenset[str] = frozenset(
    {
        "video_title",
        "duration_sec",
        "frames_requested",
        "frames_published",
        "notion_page_id",
        "notion_page_url",
    }
)


@dataclass(frozen=True)
class JobRecord:
    """The job row as a stage sees it."""

    id: UUID
    video_id: str
    youtube_url: str
    status: JobStatus
    progress: int
    video_title: str | None
    duration_sec: int | None
    notion_parent_id: str | None
    notion_page_id: str | None
    notion_page_url: str | None
    warnings: tuple[JobWarning, ...]
    enqueued_at: datetime | None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> JobRecord:
        return cls(
            id=row["id"],
            video_id=row["video_id"],
            youtube_url=row["youtube_url"],
            status=JobStatus(row["status"]),
            progress=row["progress"],
            video_title=row["video_title"],
            duration_sec=row["duration_sec"],
            notion_parent_id=row["notion_parent_id"],
            notion_page_id=row["notion_page_id"],
            notion_page_url=row["notion_page_url"],
            warnings=tuple(JobWarning.model_validate(item) for item in row["warnings"] or []),
            enqueued_at=row["enqueued_at"],
        )


@dataclass(frozen=True)
class StageOutcome:
    """What a stage hands back; the runtime makes it durable before acking (invariant 3)."""

    summary: Mapping[str, Any]  # checkpoint output: small and JSON-serialisable
    warnings: tuple[JobWarning, ...] = ()
    job_updates: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.job_updates) - JOB_UPDATE_COLUMNS
        if unknown:
            raise ValueError(f"stages may not set job columns {sorted(unknown)}")


ProgressReporter = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class StageContext:
    job: JobRecord
    message: StageMessage
    job_dir: Path  # /data/jobs/{job_id}
    settings: Settings
    log: FilteringBoundLogger
    report_progress: ProgressReporter  # fraction 0..1 of the stage's progress range


StageFn = Callable[[StageContext], Awaitable[StageOutcome]]
