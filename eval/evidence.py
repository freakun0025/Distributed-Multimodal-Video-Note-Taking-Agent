"""What a finished job left behind, for the eval scripts (ARCHITECTURE §5 artefacts, §14).

A job's folder keeps what its Notion page was built from: `synthesis.json` (the notes, captions
and picks), `frames/` (the JPEGs uploaded to the page and their manifest), `candidates/` (the
frames the model saw) and `transcript.json`. The page itself needs the user's Notion token, which
only the extension holds, so the evals read these instead; PUBLISH builds the page from exactly
these files (`worker/notion_page.py`).

Runs inside a worker container (`make check-constraints`, `make judge`): the database, the
`/data` volume, ffmpeg and the LLM settings are all there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from common.db import jobs
from common.models import FrameResult, Synthesis, Transcript, VideoMeta
from common.state import JobStatus

VIDEOS_FILE = Path(__file__).with_name("videos.json")
SUCCESS = (JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS)
FAKE_TITLE_PREFIX = "Fake "  # fakes/app.py names every fake video "Fake lecture <id>"


class OutdatedJobError(Exception):
    """The job's files were written by an older version of the notes format."""


@dataclass(frozen=True)
class EvalVideo:
    video_id: str
    kind: str
    held_out: bool


@dataclass
class JobEvidence:
    job_id: UUID
    video_id: str
    title: str | None
    status: JobStatus
    duration_sec: int | None
    frames_requested: int | None
    frames_published: int | None
    warnings: list[dict[str, Any]]
    created_at: datetime
    folder: Path
    synthesis: Synthesis | None = None
    transcript: Transcript | None = None
    meta: VideoMeta | None = None
    frames: list[FrameResult] = field(default_factory=list)
    candidates: dict[int, Path] = field(default_factory=dict)  # t_sec -> JPEG the model saw

    @property
    def published(self) -> list[FrameResult]:
        return [frame for frame in self.frames if frame.ok and frame.path]

    def caption(self, t_sec: int) -> tuple[str, str] | None:
        """(caption, reason) of the pick at `t_sec`."""
        if self.synthesis is None:
            return None
        for pick in self.synthesis.frames:
            if pick.t_sec == t_sec:
                return pick.caption, pick.reason
        return None


def eval_videos(path: Path = VIDEOS_FILE) -> list[EvalVideo]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalVideo(video_id=entry["video_id"], kind=entry["kind"], held_out=group == "held_out")
        for group in ("fixed", "held_out")
        for entry in data.get(group, [])
    ]


async def latest_successful(engine: AsyncEngine, video_ids: list[str]) -> dict[str, UUID]:
    """The newest COMPLETED / COMPLETED_WITH_WARNINGS job of each video made with the real APIs
    (the integration tests reuse real video IDs on the fakes, whose titles start "Fake ")."""
    query = (
        select(jobs.c.video_id, jobs.c.id)
        .where(
            jobs.c.video_id.in_(video_ids),
            jobs.c.status.in_([s.value for s in SUCCESS]),
            jobs.c.video_title.not_like(f"{FAKE_TITLE_PREFIX}%"),
        )
        .order_by(jobs.c.created_at)
    )
    async with engine.connect() as conn:
        rows = (await conn.execute(query)).all()
    return {row.video_id: row.id for row in rows}  # later rows win: the newest job per video


async def load(engine: AsyncEngine, jobs_dir: Path, job_id: UUID) -> JobEvidence:
    async with engine.connect() as conn:
        row = (await conn.execute(select(jobs).where(jobs.c.id == job_id))).one()
    folder = jobs_dir / str(job_id)
    evidence = JobEvidence(
        job_id=row.id,
        video_id=row.video_id,
        title=row.video_title,
        status=JobStatus(row.status),
        duration_sec=row.duration_sec,
        frames_requested=row.frames_requested,
        frames_published=row.frames_published,
        warnings=list(row.warnings or []),
        created_at=row.created_at,
        folder=folder,
    )
    if (data := _read(folder / "synthesis.json")) is not None:
        try:
            evidence.synthesis = Synthesis.model_validate(data)
        except ValidationError as exc:
            raise OutdatedJobError(
                f"job {job_id} predates the current notes format; run the video again"
            ) from exc
    if (data := _read(folder / "transcript.json")) is not None:
        evidence.transcript = Transcript.model_validate(data)
    if (data := _read(folder / "meta.json")) is not None:
        evidence.meta = VideoMeta.model_validate(data)
    if (data := _read(folder / "frames" / "manifest.json")) is not None:
        evidence.frames = [FrameResult.model_validate(item) for item in data]
    if (data := _read(folder / "candidates" / "manifest.json")) is not None:
        evidence.candidates = {
            int(item["t_sec"]): folder / item["path"] for item in data if item.get("path")
        }
    return evidence


def _read(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
