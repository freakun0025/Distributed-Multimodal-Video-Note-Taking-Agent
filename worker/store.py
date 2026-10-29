"""The worker's Postgres operations, behind a protocol so the runtime is testable (ARCHITECTURE §7).

Every status change goes through common.state.transition() (invariant 5).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy import exists, func, insert, literal, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from common.db import job_events, jobs, stage_checkpoints
from common.errors import RetryableError
from common.models import ErrorInfo
from common.state import (
    ERROR_STATUSES,
    STAGE_ORDER,
    JobStatus,
    Stage,
    next_stage,
    progress_at,
    status_before,
    status_of,
    transition,
)
from worker.context import JobRecord, StageOutcome


class JobStore(Protocol):
    async def load(self, job_id: UUID) -> JobRecord | None: ...

    async def has_checkpoint(self, job_id: UUID, stage: Stage) -> bool: ...

    async def start_stage(self, job_id: UUID, stage: Stage, attempt: int, worker: str) -> bool:
        """Enter the stage's status (first time or re-entry). False → the message is stale."""
        ...

    async def report_progress(self, job_id: UUID, stage: Stage, fraction: float) -> None: ...

    async def complete_stage(
        self, job_id: UUID, stage: Stage, attempt: int, outcome: StageOutcome, worker: str
    ) -> bool:
        """Checkpoint + outcome in one transaction (+ the final status after PUBLISH).

        False → another delivery of this stage got there first, or the job row is gone;
        nothing was written.
        """
        ...

    async def record_retry(
        self,
        job_id: UUID,
        stage: Stage,
        attempt: int,
        error: RetryableError,
        delay_s: int,
        worker: str,
    ) -> None: ...

    async def finish_with_error(
        self,
        job_id: UUID,
        status: JobStatus,
        stage: Stage,
        attempt: int,
        error: ErrorInfo,
        worker: str,
    ) -> bool: ...

    async def record_redelivery(
        self, job_id: UUID, stage: Stage, attempt: int, delivery_count: int, worker: str
    ) -> None: ...


class _SupersededError(Exception):
    """Rolls back complete_stage when the job left PUBLISHING underneath us."""


class PostgresJobStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def load(self, job_id: UUID) -> JobRecord | None:
        async with self._engine.connect() as conn:
            row = (await conn.execute(select(jobs).where(jobs.c.id == job_id))).mappings().first()
        return JobRecord.from_row(dict(row)) if row is not None else None

    async def has_checkpoint(self, job_id: UUID, stage: Stage) -> bool:
        async with self._engine.connect() as conn:
            found = await conn.scalar(
                select(
                    exists().where(
                        stage_checkpoints.c.job_id == job_id,
                        stage_checkpoints.c.stage == stage.value,
                    )
                )
            )
        return bool(found)

    async def start_stage(self, job_id: UUID, stage: Stage, attempt: int, worker: str) -> bool:
        async with self._engine.begin() as conn:
            moved = await transition(
                conn,
                job_id,
                status_of(stage),
                allowed_from={status_before(stage), status_of(stage)},
                stage=stage,
                attempt=attempt,
                detail={"worker": worker},
            )
            if moved is None:
                return False
            if stage is STAGE_ORDER[0]:
                # First claim closes the outbox window if the gateway never saved enqueued_at.
                await conn.execute(
                    update(jobs)
                    .where(jobs.c.id == job_id, jobs.c.enqueued_at.is_(None))
                    .values(enqueued_at=func.now())
                )
        return True

    async def report_progress(self, job_id: UUID, stage: Stage, fraction: float) -> None:
        status = status_of(stage)
        value = progress_at(status, fraction)
        async with self._engine.begin() as conn:
            await conn.execute(
                update(jobs)
                .where(jobs.c.id == job_id, jobs.c.status == status.value, jobs.c.progress < value)
                .values(progress=value, updated_at=func.now())
            )

    async def complete_stage(
        self, job_id: UUID, stage: Stage, attempt: int, outcome: StageOutcome, worker: str
    ) -> bool:
        try:
            async with self._engine.begin() as conn:
                # Lock the job row: it cannot vanish or change status under this transaction.
                locked = await conn.scalar(
                    select(jobs.c.id).where(jobs.c.id == job_id).with_for_update()
                )
                if locked is None:
                    return False
                inserted = await conn.scalar(
                    pg_insert(stage_checkpoints)
                    .values(job_id=job_id, stage=stage.value, output=dict(outcome.summary))
                    .on_conflict_do_nothing()
                    .returning(stage_checkpoints.c.stage)
                )
                if inserted is None:
                    return False
                await self._apply_outcome(conn, job_id, stage, attempt, outcome, worker)
                if next_stage(stage) is None:
                    await self._finish(conn, job_id, stage, attempt, worker)
        except _SupersededError:
            return False
        return True

    async def _apply_outcome(
        self,
        conn: AsyncConnection,
        job_id: UUID,
        stage: Stage,
        attempt: int,
        outcome: StageOutcome,
        worker: str,
    ) -> None:
        status = status_of(stage)
        values: dict[str, Any] = {**outcome.job_updates, "updated_at": func.now()}
        values["progress"] = progress_at(status, 1.0)
        warnings = [warning.model_dump(mode="json") for warning in outcome.warnings]
        if warnings:
            values["warnings"] = jobs.c.warnings.op("||", return_type=JSONB)(
                literal(warnings, JSONB)
            )
        await conn.execute(
            update(jobs).where(jobs.c.id == job_id, jobs.c.status == status.value).values(values)
        )
        for warning in warnings:
            await conn.execute(
                insert(job_events).values(
                    job_id=job_id,
                    kind="WARNING",
                    stage=stage.value,
                    attempt=attempt,
                    detail={**warning, "worker": worker},
                )
            )

    async def _finish(
        self, conn: AsyncConnection, job_id: UUID, stage: Stage, attempt: int, worker: str
    ) -> None:
        warnings = await conn.scalar(select(jobs.c.warnings).where(jobs.c.id == job_id))
        final = JobStatus.COMPLETED_WITH_WARNINGS if warnings else JobStatus.COMPLETED
        moved = await transition(
            conn,
            job_id,
            final,
            allowed_from={status_of(stage)},
            stage=stage,
            attempt=attempt,
            detail={"worker": worker},
        )
        if moved is None:
            raise _SupersededError

    async def record_retry(
        self,
        job_id: UUID,
        stage: Stage,
        attempt: int,
        error: RetryableError,
        delay_s: int,
        worker: str,
    ) -> None:
        async with self._engine.begin() as conn:
            waiting = await conn.scalar(
                update(jobs)
                .where(jobs.c.id == job_id, jobs.c.status == status_of(stage).value)
                .values(
                    waiting_until=func.now() + timedelta(seconds=delay_s),
                    waiting_reason=error.waiting_reason,
                    updated_at=func.now(),
                )
                .returning(jobs.c.id)
            )
            if waiting is None:
                return  # job finished or vanished meanwhile: nothing to show as waiting
            await conn.execute(
                insert(job_events).values(
                    job_id=job_id,
                    kind="RETRY_SCHEDULED",
                    stage=stage.value,
                    attempt=attempt,
                    detail={**error.to_info(), "delay_s": delay_s, "worker": worker},
                )
            )

    async def finish_with_error(
        self,
        job_id: UUID,
        status: JobStatus,
        stage: Stage,
        attempt: int,
        error: ErrorInfo,
        worker: str,
    ) -> bool:
        if status not in ERROR_STATUSES:
            raise ValueError(f"{status} is not an error status")
        async with self._engine.begin() as conn:
            moved = await transition(
                conn,
                job_id,
                status,
                error=error,
                stage=stage,
                attempt=attempt,
                detail={"worker": worker},
            )
            if moved is not None and status is JobStatus.DEAD_LETTERED:
                await conn.execute(
                    insert(job_events).values(
                        job_id=job_id,
                        kind="DEAD_LETTERED",
                        stage=stage.value,
                        attempt=attempt,
                        detail={**error.model_dump(mode="json"), "worker": worker},
                    )
                )
        return moved is not None

    async def record_redelivery(
        self, job_id: UUID, stage: Stage, attempt: int, delivery_count: int, worker: str
    ) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                insert(job_events).values(
                    job_id=job_id,
                    kind="REDELIVERED",
                    stage=stage.value,
                    attempt=attempt,
                    detail={"delivery_count": delivery_count, "worker": worker},
                )
            )
