"""Test doubles shared by unit tests."""

from collections.abc import Sequence
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


class MemoryModelHealth:
    """In-process ModelHealth: the bench/strike rules of worker.llm.health without Redis."""

    def __init__(self, *, strikes: int = 2, quota_reset_s: float = 3600) -> None:
        self.strikes_needed = strikes
        self.quota_reset_s = quota_reset_s
        self.bench: dict[str, float] = {}  # model -> seconds benched
        self.strikes: dict[str, int] = {}

    async def benched(self, models: Sequence[str]) -> dict[str, float]:
        return {model: self.bench[model] for model in models if model in self.bench}

    async def quota_spent(self, model: str) -> float:
        self.bench[model] = self.quota_reset_s
        return self.quota_reset_s

    async def overloaded(self, model: str) -> bool:
        self.strikes[model] = self.strikes.get(model, 0) + 1
        if self.strikes[model] >= self.strikes_needed:
            self.bench[model] = 600.0
            del self.strikes[model]
            return True
        return False

    async def answered(self, model: str) -> None:
        self.strikes.pop(model, None)


def make_context(
    tmp_path: Path,
    stage: Stage,
    *,
    progress: list[float] | None = None,
    limiter: FakeLimiter | None = None,
    model_health: MemoryModelHealth | None = None,
    attempt: int = 1,
    duration_sec: int | None = None,
    saved_pages: list[tuple[str, str]] | None = None,
    notion_page: tuple[str, str] | None = None,
    **settings: Any,
) -> StageContext:
    """A StageContext for one stage of a fresh job, artefacts under `tmp_path`. `saved_pages`
    collects what the stage saves through `save_notion_page`; `notion_page` is (id, url) already
    saved on the job."""
    job_id = uuid4()
    reported = progress if progress is not None else []
    saved = saved_pages if saved_pages is not None else []

    async def report(fraction: float) -> None:
        reported.append(fraction)

    async def save_page(page_id: str, url: str) -> None:
        saved.append((page_id, url))

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
            notion_page_id=notion_page[0] if notion_page else None,
            notion_page_url=notion_page[1] if notion_page else None,
            warnings=(),
            enqueued_at=None,
        ),
        message=StageMessage(job_id=job_id, stage=stage, attempt=attempt, notion_token_enc="enc"),
        job_dir=tmp_path / str(job_id),
        settings=Settings(_env_file=None, data_dir=tmp_path, **settings),  # type: ignore[call-arg]
        log=get_logger(),
        report_progress=report,
        limiter=limiter or FakeLimiter(),
        model_health=model_health or MemoryModelHealth(),
        save_notion_page=save_page,
    )
