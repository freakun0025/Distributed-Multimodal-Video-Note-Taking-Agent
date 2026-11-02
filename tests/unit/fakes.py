"""Test doubles shared by unit tests."""

from pathlib import Path
from typing import Any
from uuid import uuid4

from common.config import Settings
from common.errors import ExternalApi
from common.logging import get_logger
from common.models import StageMessage
from common.state import JobStatus, Stage
from worker.context import JobRecord, StageContext

VIDEO_ID = "dQw4w9WgXcQ"


class FakeLimiter:
    """Grants every acquire at once and records it, so tests can assert a call was limited."""

    def __init__(self) -> None:
        self.calls: list[tuple[ExternalApi, float]] = []

    async def acquire(self, api: ExternalApi, cost: float = 1) -> float:
        self.calls.append((api, cost))
        return 0.0


def make_context(
    tmp_path: Path,
    stage: Stage,
    *,
    progress: list[float] | None = None,
    limiter: FakeLimiter | None = None,
    attempt: int = 1,
    duration_sec: int | None = None,
    **settings: Any,
) -> StageContext:
    """A StageContext for one stage of a fresh job, artefacts under `tmp_path`."""
    job_id = uuid4()
    reported = progress if progress is not None else []

    async def report(fraction: float) -> None:
        reported.append(fraction)

    return StageContext(
        job=JobRecord(
            id=job_id,
            video_id=VIDEO_ID,
            youtube_url=f"https://www.youtube.com/watch?v={VIDEO_ID}",
            status=JobStatus.QUEUED,
            progress=0,
            video_title=None,
            duration_sec=duration_sec,
            notion_parent_id=None,
            notion_page_id=None,
            notion_page_url=None,
            warnings=(),
            enqueued_at=None,
        ),
        message=StageMessage(job_id=job_id, stage=stage, attempt=attempt, notion_token_enc="enc"),
        job_dir=tmp_path / str(job_id),
        settings=Settings(_env_file=None, data_dir=tmp_path, **settings),  # type: ignore[call-arg]
        log=get_logger(),
        report_progress=report,
        limiter=limiter or FakeLimiter(),
    )
