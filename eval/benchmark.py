"""End-to-end latency per stage, measured from the job's own events (Spec §9; O-7).

    make bench-report                  the newest successful job of each video in eval/videos.json
    make bench-report ids="<id> <id>"  these jobs

For each job: the wait in the queue (submitted -> a worker claims it), each stage from the moment
the job entered its status to the moment it left it (retries and their delays included, so a
503 or a refused download shows up where it happened; with more jobs than workers, also the
wait for a free worker), and the total from submission to the end. Every number is read from
`job_events` (the database's clock), never estimated.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from common.config import get_settings
from common.db import create_engine, job_events, jobs
from common.state import JobStatus
from eval.evidence import eval_videos, latest_successful
from worker.llm.chunking import clock

STAGES = (
    JobStatus.TRANSCRIBING,
    JobStatus.SUMMARIZING,
    JobStatus.EXTRACTING_FRAMES,
    JobStatus.PUBLISHING,
)


@dataclass(frozen=True)
class Timing:
    job_id: UUID
    title: str
    duration_sec: int | None
    status: str
    queued_s: float
    stages_s: dict[JobStatus, float]
    retries: int
    total_s: float


async def timing(engine: AsyncEngine, job_id: UUID) -> Timing:
    async with engine.connect() as conn:
        job = (await conn.execute(select(jobs).where(jobs.c.id == job_id))).one()
        events = (
            await conn.execute(
                select(job_events.c.at, job_events.c.kind, job_events.c.to_status)
                .where(job_events.c.job_id == job_id)
                .order_by(job_events.c.id)
            )
        ).all()
    entered: dict[JobStatus, datetime] = {}
    left: dict[JobStatus, datetime] = {}
    current: JobStatus | None = None
    for event in events:
        if event.kind != "TRANSITION" or event.to_status is None:
            continue
        status = JobStatus(event.to_status)
        if status is current:
            continue  # a retry or redelivery of the same stage
        if current is not None:
            left[current] = event.at
        entered.setdefault(status, event.at)
        current = status
    end: datetime = job.finished_at or job.updated_at
    first = entered.get(JobStatus.TRANSCRIBING, end)
    return Timing(
        job_id=job.id,
        title=job.video_title or job.video_id,
        duration_sec=job.duration_sec,
        status=job.status,
        queued_s=(first - job.created_at).total_seconds(),
        stages_s={
            stage: (left.get(stage, end) - entered[stage]).total_seconds()
            for stage in STAGES
            if stage in entered
        },
        retries=sum(1 for event in events if event.kind == "RETRY_SCHEDULED"),
        total_s=(end - job.created_at).total_seconds(),
    )


def _s(seconds: float) -> str:
    return f"{seconds:6.1f}"


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    try:
        if args.jobs:
            job_ids = [UUID(job) for job in args.jobs]
        else:
            videos = [video.video_id for video in eval_videos()]
            newest = await latest_successful(engine, videos)
            job_ids = [newest[v] for v in videos if v in newest]
        rows = [await timing(engine, job_id) for job_id in job_ids]
        print(
            f"{'video':40} {'length':>6} {'queued':>6} {'transc':>6} {'summar':>6} "
            f"{'frames':>6} {'publis':>6} {'total':>6} retries"
        )
        for row in rows:
            cells = [_s(row.stages_s.get(stage, 0.0)) for stage in STAGES]
            print(
                f"{row.title[:40]:40} {clock(row.duration_sec or 0):>6} {_s(row.queued_s)} "
                f"{' '.join(cells)} {_s(row.total_s)} {row.retries:>7}"
            )
        print(
            "\nSeconds. A stage runs from entering its status to leaving it, retries included; a"
            "\njob keeps the status until a worker claims the next stage, so with more jobs than"
            "\nworkers a stage's time also holds the wait for a free worker."
        )
        return 0
    finally:
        await engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("jobs", nargs="*", help="job IDs (default: the videos in videos.json)")
    return parser


if __name__ == "__main__":
    sys.exit(asyncio.run(main(_parser().parse_args())))
