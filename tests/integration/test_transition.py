"""transition() against the real Postgres schema (ARCHITECTURE §7, invariant 5)."""

import asyncio
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from common.db import job_events, jobs
from common.errors import ErrorCode
from common.models import ErrorInfo
from common.state import JobStatus, Stage, transition
from tests.integration.conftest import JobRows, advance

pytestmark = pytest.mark.integration

S = JobStatus
ERROR = ErrorInfo(code=ErrorCode.INTERNAL_ERROR, message="boom")


async def _job(engine: AsyncEngine, job_id: UUID) -> dict[str, object]:
    async with engine.connect() as conn:
        row = (await conn.execute(select(jobs).where(jobs.c.id == job_id))).mappings().one()
    return dict(row)


async def _events(engine: AsyncEngine, job_id: UUID) -> list[dict[str, object]]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            select(job_events).where(job_events.c.job_id == job_id).order_by(job_events.c.id)
        )
        return [dict(row) for row in rows.mappings()]


async def test_first_stage_moves_queued_job_and_records_the_event(
    engine: AsyncEngine, job_rows: JobRows
) -> None:
    job_id = await job_rows.create()
    async with engine.begin() as conn:
        moved = await transition(conn, job_id, S.TRANSCRIBING, stage=Stage.TRANSCRIBE, attempt=1)
    assert moved is S.QUEUED
    job = await _job(engine, job_id)
    assert job["status"] == "TRANSCRIBING"
    assert job["progress"] == 5
    [event] = await _events(engine, job_id)
    assert event["kind"] == "TRANSITION"
    assert (event["from_status"], event["to_status"]) == ("QUEUED", "TRANSCRIBING")
    assert (event["stage"], event["attempt"]) == ("TRANSCRIBE", 1)


async def test_redelivery_re_enters_its_own_stage_and_keeps_progress(
    engine: AsyncEngine, job_rows: JobRows
) -> None:
    job_id = await job_rows.create()
    async with engine.begin() as conn:
        await advance(conn, job_id, S.SUMMARIZING)
        await transition(conn, job_id, S.SUMMARIZING, progress=35)
        moved = await transition(conn, job_id, S.SUMMARIZING, stage=Stage.SUMMARIZE, attempt=2)
    assert moved is S.SUMMARIZING
    assert (await _job(engine, job_id))["progress"] == 35


async def test_disallowed_transition_changes_nothing_and_writes_no_event(
    engine: AsyncEngine, job_rows: JobRows
) -> None:
    job_id = await job_rows.create()
    async with engine.begin() as conn:
        assert await transition(conn, job_id, S.PUBLISHING) is None
        assert await transition(conn, job_id, S.COMPLETED) is None
    assert (await _job(engine, job_id))["status"] == "QUEUED"
    assert await _events(engine, job_id) == []


async def test_failure_records_error_and_finish_time_and_clears_waiting(
    engine: AsyncEngine, job_rows: JobRows
) -> None:
    job_id = await job_rows.create(waiting_reason="RATE_LIMITED:llm")
    async with engine.begin() as conn:
        await advance(conn, job_id, S.TRANSCRIBING)
        moved = await transition(conn, job_id, S.FAILED, error=ERROR)
    assert moved is S.TRANSCRIBING
    job = await _job(engine, job_id)
    assert job["status"] == "FAILED"
    assert job["error"] == {"code": "INTERNAL_ERROR", "message": "boom"}
    assert job["finished_at"] is not None
    assert job["progress"] == 5  # FAILED keeps the last progress
    assert job["waiting_reason"] is None
    assert (await _events(engine, job_id))[-1]["detail"] == {"error": job["error"]}


async def test_a_terminal_job_never_moves_again(engine: AsyncEngine, job_rows: JobRows) -> None:
    job_id = await job_rows.create()
    async with engine.begin() as conn:
        await advance(conn, job_id, S.PUBLISHING)
        assert await transition(conn, job_id, S.COMPLETED) is S.PUBLISHING
        assert await transition(conn, job_id, S.FAILED, error=ERROR) is None
        assert await transition(conn, job_id, S.PUBLISHING) is None
    job = await _job(engine, job_id)
    assert (job["status"], job["progress"]) == ("COMPLETED", 100)


async def test_concurrent_finishers_exactly_one_wins(
    engine: AsyncEngine, job_rows: JobRows
) -> None:
    job_id = await job_rows.create()
    async with engine.begin() as conn:
        await advance(conn, job_id, S.PUBLISHING)

    async def finish(to: JobStatus) -> JobStatus | None:
        async with engine.begin() as conn:
            return await transition(conn, job_id, to, error=ERROR if to is S.FAILED else None)

    results = await asyncio.gather(finish(S.COMPLETED), finish(S.FAILED))
    assert sorted(result is not None for result in results) == [False, True]
    assert len([e for e in await _events(engine, job_id) if e["from_status"] == "PUBLISHING"]) == 1


async def test_unknown_job_is_not_moved(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        assert await transition(conn, uuid4(), S.TRANSCRIBING) is None
