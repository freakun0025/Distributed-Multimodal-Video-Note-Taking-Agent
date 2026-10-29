"""PostgresJobStore against the real schema: the SQL behind the runtime's decisions."""

from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine

from common.db import job_events, jobs, stage_checkpoints
from common.errors import ErrorCode, ExternalApi, RetryableError, WarningCode
from common.models import ErrorInfo, JobWarning
from common.state import JobStatus, Stage
from tests.integration.conftest import JobRows, advance
from worker.context import StageOutcome
from worker.store import PostgresJobStore

pytestmark = pytest.mark.integration

WARNING = JobWarning(code=WarningCode.FRAME_EXTRACTION_FAILED, timestamp_sec=461)


async def _row(engine: AsyncEngine, job_id: UUID) -> dict[str, Any]:
    async with engine.connect() as conn:
        return dict((await conn.execute(select(jobs).where(jobs.c.id == job_id))).mappings().one())


async def _event_kinds(engine: AsyncEngine, job_id: UUID) -> list[str]:
    async with engine.connect() as conn:
        rows = await conn.scalars(
            select(job_events.c.kind).where(job_events.c.job_id == job_id).order_by(job_events.c.id)
        )
        return list(rows.all())


async def test_first_stage_claim_sets_enqueued_at(engine: AsyncEngine, job_rows: JobRows) -> None:
    store = PostgresJobStore(engine)
    job_id = await job_rows.create()
    assert await store.start_stage(job_id, Stage.TRANSCRIBE, 1, "w1")
    row = await _row(engine, job_id)
    assert (row["status"], row["enqueued_at"] is not None) == ("TRANSCRIBING", True)


async def test_a_stage_cannot_start_out_of_order(engine: AsyncEngine, job_rows: JobRows) -> None:
    store = PostgresJobStore(engine)
    job_id = await job_rows.create()
    assert not await store.start_stage(job_id, Stage.SUMMARIZE, 1, "w1")
    assert (await _row(engine, job_id))["status"] == "QUEUED"


async def test_progress_only_rises_and_only_within_the_current_stage(
    engine: AsyncEngine, job_rows: JobRows
) -> None:
    store = PostgresJobStore(engine)
    job_id = await job_rows.create()
    async with engine.begin() as conn:
        await advance(conn, job_id, JobStatus.SUMMARIZING)
    await store.report_progress(job_id, Stage.SUMMARIZE, 0.5)
    await store.report_progress(job_id, Stage.SUMMARIZE, 0.2)  # lower: ignored
    await store.report_progress(job_id, Stage.EXTRACT_FRAMES, 1.0)  # other stage: ignored
    assert (await _row(engine, job_id))["progress"] == 35


async def test_checkpoint_is_written_once_with_warnings_and_job_updates(
    engine: AsyncEngine, job_rows: JobRows
) -> None:
    store = PostgresJobStore(engine)
    job_id = await job_rows.create()
    async with engine.begin() as conn:
        await advance(conn, job_id, JobStatus.EXTRACTING_FRAMES)
    outcome = StageOutcome(
        summary={"frames_ok": 4}, warnings=(WARNING,), job_updates={"frames_requested": 5}
    )
    assert await store.complete_stage(job_id, Stage.EXTRACT_FRAMES, 1, outcome, "w1")
    assert not await store.complete_stage(job_id, Stage.EXTRACT_FRAMES, 1, outcome, "w2")

    row = await _row(engine, job_id)
    assert row["warnings"] == [{"code": "FRAME_EXTRACTION_FAILED", "timestamp_sec": 461}]
    assert (row["frames_requested"], row["progress"]) == (5, 80)
    assert await store.has_checkpoint(job_id, Stage.EXTRACT_FRAMES)
    assert (await _event_kinds(engine, job_id)).count("WARNING") == 1


@pytest.mark.parametrize(
    ("warnings", "final"),
    [((), "COMPLETED"), ((WARNING,), "COMPLETED_WITH_WARNINGS")],
)
async def test_publish_checkpoint_and_final_status_commit_together(
    engine: AsyncEngine, job_rows: JobRows, warnings: tuple[JobWarning, ...], final: str
) -> None:
    store = PostgresJobStore(engine)
    job_id = await job_rows.create()
    async with engine.begin() as conn:
        await advance(conn, job_id, JobStatus.PUBLISHING)
    outcome = StageOutcome(summary={"blocks": 12}, warnings=warnings)
    assert await store.complete_stage(job_id, Stage.PUBLISH, 1, outcome, "w1")
    row = await _row(engine, job_id)
    assert (row["status"], row["progress"]) == (final, 100)
    assert row["finished_at"] is not None


async def test_publish_checkpoint_rolls_back_if_the_job_left_publishing(
    engine: AsyncEngine, job_rows: JobRows
) -> None:
    store = PostgresJobStore(engine)
    job_id = await job_rows.create()
    async with engine.begin() as conn:
        await advance(conn, job_id, JobStatus.PUBLISHING)
    error = ErrorInfo(code=ErrorCode.NOTION_UNAUTHORIZED, message="revoked")
    await store.finish_with_error(job_id, JobStatus.FAILED, Stage.PUBLISH, 1, error, "w2")
    assert not await store.complete_stage(job_id, Stage.PUBLISH, 1, StageOutcome(summary={}), "w1")
    assert not await store.has_checkpoint(job_id, Stage.PUBLISH)
    assert (await _row(engine, job_id))["status"] == "FAILED"


async def test_a_job_deleted_mid_stage_is_handled_quietly(
    engine: AsyncEngine, job_rows: JobRows
) -> None:
    store = PostgresJobStore(engine)
    job_id = await job_rows.create()
    async with engine.begin() as conn:
        await advance(conn, job_id, JobStatus.TRANSCRIBING)
        await conn.execute(delete(jobs).where(jobs.c.id == job_id))
    outcome = StageOutcome(summary={})
    assert not await store.complete_stage(job_id, Stage.TRANSCRIBE, 1, outcome, "w1")
    error = RetryableError(ErrorCode.UPSTREAM_ERROR, "5xx")
    await store.record_retry(job_id, Stage.TRANSCRIBE, 1, error, 10, "w1")  # no FK error
    assert await store.load(job_id) is None
    async with engine.connect() as conn:
        found = await conn.scalar(
            select(stage_checkpoints.c.stage).where(stage_checkpoints.c.job_id == job_id)
        )
    assert found is None


async def test_retry_shows_waiting_until_the_stage_is_re_entered(
    engine: AsyncEngine, job_rows: JobRows
) -> None:
    store = PostgresJobStore(engine)
    job_id = await job_rows.create()
    async with engine.begin() as conn:
        await advance(conn, job_id, JobStatus.SUMMARIZING)
    error = RetryableError(ErrorCode.RATE_LIMITED, "429", api=ExternalApi.LLM)
    await store.record_retry(job_id, Stage.SUMMARIZE, 1, error, 30, "w1")
    row = await _row(engine, job_id)
    assert row["waiting_reason"] == "RATE_LIMITED:llm"
    assert row["waiting_until"] > row["updated_at"]
    assert await store.start_stage(job_id, Stage.SUMMARIZE, 2, "w2")
    assert (await _row(engine, job_id))["waiting_reason"] is None


async def test_dead_lettering_records_status_error_and_event(
    engine: AsyncEngine, job_rows: JobRows
) -> None:
    store = PostgresJobStore(engine)
    job_id = await job_rows.create()
    async with engine.begin() as conn:
        await advance(conn, job_id, JobStatus.TRANSCRIBING)
    error = ErrorInfo(code=ErrorCode.VIDEO_UNAVAILABLE, message="private video")
    assert await store.finish_with_error(
        job_id, JobStatus.DEAD_LETTERED, Stage.TRANSCRIBE, 5, error, "w1"
    )
    row = await _row(engine, job_id)
    assert row["status"] == "DEAD_LETTERED"
    assert row["error"] == {"code": "VIDEO_UNAVAILABLE", "message": "private video"}
    assert (await _event_kinds(engine, job_id))[-2:] == ["TRANSITION", "DEAD_LETTERED"]


async def test_loading_an_unknown_job_returns_none(engine: AsyncEngine) -> None:
    assert await PostgresJobStore(engine).load(uuid4()) is None
