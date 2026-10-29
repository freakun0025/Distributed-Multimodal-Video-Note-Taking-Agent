"""Outbox sweeper: fails QUEUED jobs never published after OUTBOX_TIMEOUT_S (ARCHITECTURE §10).

The token is never persisted, so a job whose first message was never confirmed cannot be
re-published; it is marked FAILED / ENQUEUE_FAILED instead and the user retries.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from common.db import jobs
from common.errors import ErrorCode
from common.logging import get_logger
from common.models import ErrorInfo
from common.state import JobStatus, transition

log = get_logger()

_NEVER_ENQUEUED = ErrorInfo(
    code=ErrorCode.ENQUEUE_FAILED, message="job was never handed to the broker; submit it again"
)


async def sweep_once(engine: AsyncEngine, timeout_s: float) -> int:
    """Fail every stale never-enqueued job; return how many were failed."""
    failed = 0
    async with engine.begin() as conn:
        stale = await conn.scalars(
            select(jobs.c.id)
            .where(
                jobs.c.status == JobStatus.QUEUED.value,
                jobs.c.enqueued_at.is_(None),
                jobs.c.created_at < func.now() - timedelta(seconds=timeout_s),
            )
            .with_for_update(skip_locked=True)
        )
        for job_id in stale.all():
            moved = await transition(
                conn,
                job_id,
                JobStatus.FAILED,
                allowed_from={JobStatus.QUEUED},
                error=_NEVER_ENQUEUED,
                detail={"reason": "outbox_timeout"},
                extra_where=jobs.c.enqueued_at.is_(None),
            )
            if moved is not None:
                failed += 1
                log.warning("outbox_job_failed", job_id=str(job_id))
    return failed


async def run_sweeper(engine: AsyncEngine, interval_s: float, timeout_s: float) -> None:
    """Background task for the gateway's lifetime; one failed sweep never stops the next."""
    while True:
        try:
            await sweep_once(engine, timeout_s)
        except Exception as exc:
            log.warning("outbox_sweep_failed", error=type(exc).__name__)
        await asyncio.sleep(interval_s)
