"""SIGKILL the worker running a stage, then follow the job (Spec §6 worker crashes, §9 recovery).

    make kill-worker stage=EXTRACT_FRAMES
    uv run python -m chaos.kill_worker --stage PUBLISH --job <job id>

Waits until a job is running the stage (the newest such job, or `--job`), finds the worker running
it from the job's events (a worker is named by its container hostname, the short container ID),
kills that container with `docker kill --signal KILL`, then reports which worker the broker gave
the message to and when the job finished. The killed container is started again at the end
(`--leave-dead` keeps it down; Compose's restart policy does not restart a container killed by
hand).

Runs on the host: Postgres on the port Compose publishes (`CHAOS_DATABASE_URL` overrides it) and
the docker CLI. Times are the database's clock, so they match the job's events.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncEngine

from common.db import create_engine, job_events, jobs
from common.state import STAGE_ORDER, JobStatus, Stage, is_terminal, status_of

DATABASE_URL = os.environ.get(
    "CHAOS_DATABASE_URL", "postgresql+asyncpg://vna:vna@localhost:5433/vna"
)
POLL_S = 0.1


@dataclass(frozen=True)
class Running:
    """A job running a stage right now, and the worker container running it."""

    job_id: UUID
    worker: str


@dataclass(frozen=True)
class Recovery:
    killed_at: datetime
    redelivered_at: datetime | None
    redelivered_to: str | None
    finished_at: datetime | None
    status: JobStatus

    @property
    def reassigned_after_s(self) -> float | None:
        if self.redelivered_at is None:
            return None
        return (self.redelivered_at - self.killed_at).total_seconds()

    @property
    def finished_after_s(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.killed_at).total_seconds()


async def find_running(
    engine: AsyncEngine, stage: Stage, job_id: UUID | None = None
) -> Running | None:
    """The newest job in the stage's status that is not waiting in a delay queue, with the
    worker of its latest entry into that status (first run, retry or redelivery)."""
    status = status_of(stage)
    query = select(jobs.c.id).where(
        jobs.c.status == status.value,
        or_(jobs.c.waiting_until.is_(None), jobs.c.waiting_until <= func.now()),
    )
    if job_id is not None:
        query = query.where(jobs.c.id == job_id)
    async with engine.connect() as conn:
        found = await conn.scalar(query.order_by(jobs.c.updated_at.desc()).limit(1))
        if found is None:
            return None
        detail: dict[str, Any] | None = await conn.scalar(
            select(job_events.c.detail)
            .where(
                job_events.c.job_id == found,
                job_events.c.kind == "TRANSITION",
                job_events.c.to_status == status.value,
            )
            .order_by(job_events.c.id.desc())
            .limit(1)
        )
    worker = (detail or {}).get("worker")
    return Running(found, str(worker)) if worker else None


async def wait_running(
    engine: AsyncEngine, stage: Stage, job_id: UUID | None, timeout_s: float
) -> Running:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if (running := await find_running(engine, stage, job_id)) is not None:
            return running
        await asyncio.sleep(POLL_S)
    raise TimeoutError(f"no job entered {status_of(stage)} within {timeout_s:g} s")


async def docker(*args: str) -> None:
    process = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.DEVNULL
    )
    if await process.wait() != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed")


async def kill(engine: AsyncEngine, worker: str) -> datetime:
    """SIGKILL the worker's container; returns the database time just before the kill.

    Taken before, not after: the broker can hand the message to another worker, and that worker
    log the redelivery, before `docker kill` even returns. Recovery times measured from here
    include the kill command itself, so they are an upper bound.
    """
    async with engine.connect() as conn:
        before = await conn.scalar(select(func.clock_timestamp()))
    if not isinstance(before, datetime):
        raise TypeError(f"clock_timestamp() returned {before!r}")
    await docker("kill", "--signal", "KILL", worker)
    return before


async def follow(
    engine: AsyncEngine, job_id: UUID, stage: Stage, killed_at: datetime, timeout_s: float
) -> Recovery:
    """Wait for the job to end; note when, and to which worker, the killed stage was redelivered."""
    deadline = time.monotonic() + timeout_s
    redelivered: tuple[datetime, str] | None = None
    while True:
        async with engine.connect() as conn:
            if redelivered is None:
                event = (
                    await conn.execute(
                        select(job_events.c.at, job_events.c.detail)
                        .where(
                            job_events.c.job_id == job_id,
                            job_events.c.kind == "REDELIVERED",
                            job_events.c.stage == stage.value,
                            job_events.c.at >= killed_at,
                        )
                        .order_by(job_events.c.id)
                        .limit(1)
                    )
                ).first()
                if event is not None:
                    redelivered = (event.at, str((event.detail or {}).get("worker")))
            row = (
                await conn.execute(
                    select(jobs.c.status, jobs.c.finished_at).where(jobs.c.id == job_id)
                )
            ).one()
        status = JobStatus(row.status)
        if is_terminal(status) or time.monotonic() >= deadline:
            return Recovery(
                killed_at=killed_at,
                redelivered_at=redelivered[0] if redelivered else None,
                redelivered_to=redelivered[1] if redelivered else None,
                finished_at=row.finished_at,
                status=status,
            )
        await asyncio.sleep(POLL_S)


def _say(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


async def main(args: argparse.Namespace) -> int:
    stage = Stage(args.stage)
    engine = create_engine(args.database_url)
    try:
        _say(f"Waiting up to {args.wait:g} s for a job running {stage.value} ...")
        running = await wait_running(engine, stage, args.job, args.wait)
        killed_at = await kill(engine, running.worker)
        _say(f"Killed worker {running.worker} (SIGKILL) in {stage.value} of job {running.job_id}")
        try:
            recovery = await follow(engine, running.job_id, stage, killed_at, args.timeout)
        finally:
            if not args.leave_dead:
                await docker("start", running.worker)
                _say(f"Started {running.worker} again")
        if recovery.reassigned_after_s is not None:
            _say(
                f"Redelivered to worker {recovery.redelivered_to} "
                f"{recovery.reassigned_after_s:.1f} s after the kill"
            )
        else:
            _say("No redelivery seen")
        if recovery.finished_after_s is not None:
            _say(f"Job {recovery.status.value} {recovery.finished_after_s:.1f} s after the kill")
            return 0
        _say(f"Job still {recovery.status.value} after {args.timeout:g} s")
        return 1
    finally:
        await engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", required=True, choices=[s.value for s in STAGE_ORDER])
    parser.add_argument("--job", type=UUID, help="this job only (default: the newest in the stage)")
    parser.add_argument("--wait", type=float, default=600, help="seconds to wait for the stage")
    parser.add_argument("--timeout", type=float, default=900, help="seconds to follow the job")
    parser.add_argument("--leave-dead", action="store_true", help="do not restart the container")
    parser.add_argument("--database-url", default=DATABASE_URL)
    return parser


if __name__ == "__main__":
    sys.exit(asyncio.run(main(_parser().parse_args())))
