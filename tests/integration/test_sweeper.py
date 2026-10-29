"""Outbox sweeper against the real schema (ARCHITECTURE §10 step 6)."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from common.db import jobs
from common.state import JobStatus
from gateway.sweeper import sweep_once
from tests.integration.conftest import JobRows, advance

pytestmark = pytest.mark.integration

TIMEOUT_S = 3600


async def test_only_stale_never_enqueued_queued_jobs_are_failed(
    engine: AsyncEngine, job_rows: JobRows
) -> None:
    old = datetime.now(UTC) - timedelta(seconds=TIMEOUT_S * 2)
    stale = await job_rows.create(created_at=old)
    fresh = await job_rows.create()
    enqueued = await job_rows.create(created_at=old, enqueued_at=old)
    running = await job_rows.create(created_at=old)
    async with engine.begin() as conn:
        await advance(conn, running, JobStatus.TRANSCRIBING)

    await sweep_once(engine, TIMEOUT_S)

    async with engine.connect() as conn:
        rows = await conn.execute(
            select(jobs.c.id, jobs.c.status, jobs.c.error).where(
                jobs.c.id.in_([stale, fresh, enqueued, running])
            )
        )
        by_id = {row.id: row for row in rows}
    assert by_id[stale].status == "FAILED"
    assert by_id[stale].error["code"] == "ENQUEUE_FAILED"
    assert by_id[fresh].status == "QUEUED"
    assert by_id[enqueued].status == "QUEUED"
    assert by_id[running].status == "TRANSCRIBING"


async def test_a_second_sweep_finds_nothing_more_to_fail(
    engine: AsyncEngine, job_rows: JobRows
) -> None:
    old = datetime.now(UTC) - timedelta(seconds=TIMEOUT_S * 2)
    job_id = await job_rows.create(created_at=old)
    # The running gateway sweeps too, so either sweeper may be the one that fails this row.
    await sweep_once(engine, TIMEOUT_S)
    async with engine.connect() as conn:
        assert await conn.scalar(select(jobs.c.status).where(jobs.c.id == job_id)) == "FAILED"
    assert await sweep_once(engine, TIMEOUT_S) == 0
