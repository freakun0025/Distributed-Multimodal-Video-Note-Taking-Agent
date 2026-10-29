"""Job statuses, pipeline stages, the transition table and `transition()` (ARCHITECTURE §5, §7).

`transition()` is the only writer of `jobs.status` (invariant 5).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import func, insert, select, update

if TYPE_CHECKING:
    from sqlalchemy import ColumnElement
    from sqlalchemy.ext.asyncio import AsyncConnection

    from common.models import ErrorInfo


class JobStatus(StrEnum):
    """Exact status strings from Spec §4. They are graded: never rename."""

    QUEUED = "QUEUED"
    TRANSCRIBING = "TRANSCRIBING"
    SUMMARIZING = "SUMMARIZING"
    EXTRACTING_FRAMES = "EXTRACTING_FRAMES"
    PUBLISHING = "PUBLISHING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
    FAILED = "FAILED"
    DEAD_LETTERED = "DEAD_LETTERED"


class Stage(StrEnum):
    """One broker message per stage (ARCHITECTURE §6.3)."""

    TRANSCRIBE = "TRANSCRIBE"
    SUMMARIZE = "SUMMARIZE"
    EXTRACT_FRAMES = "EXTRACT_FRAMES"
    PUBLISH = "PUBLISH"


STAGE_ORDER: tuple[Stage, ...] = (
    Stage.TRANSCRIBE,
    Stage.SUMMARIZE,
    Stage.EXTRACT_FRAMES,
    Stage.PUBLISH,
)

STAGE_STATUS: dict[Stage, JobStatus] = {
    Stage.TRANSCRIBE: JobStatus.TRANSCRIBING,
    Stage.SUMMARIZE: JobStatus.SUMMARIZING,
    Stage.EXTRACT_FRAMES: JobStatus.EXTRACTING_FRAMES,
    Stage.PUBLISH: JobStatus.PUBLISHING,
}

TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.COMPLETED_WITH_WARNINGS,
        JobStatus.FAILED,
        JobStatus.DEAD_LETTERED,
    }
)

NON_TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(JobStatus) - TERMINAL_STATUSES

# target status -> statuses it may be entered from. Stage statuses allow a self-transition so a
# redelivered message can re-enter its own stage (ARCHITECTURE §7). QUEUED is only ever inserted.
ALLOWED_FROM: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset(),
    JobStatus.TRANSCRIBING: frozenset({JobStatus.QUEUED, JobStatus.TRANSCRIBING}),
    JobStatus.SUMMARIZING: frozenset({JobStatus.TRANSCRIBING, JobStatus.SUMMARIZING}),
    JobStatus.EXTRACTING_FRAMES: frozenset({JobStatus.SUMMARIZING, JobStatus.EXTRACTING_FRAMES}),
    JobStatus.PUBLISHING: frozenset({JobStatus.EXTRACTING_FRAMES, JobStatus.PUBLISHING}),
    JobStatus.COMPLETED: frozenset({JobStatus.PUBLISHING}),
    JobStatus.COMPLETED_WITH_WARNINGS: frozenset({JobStatus.PUBLISHING}),
    JobStatus.FAILED: NON_TERMINAL_STATUSES,
    JobStatus.DEAD_LETTERED: NON_TERMINAL_STATUSES,
}

# (start, end) progress percentage while in each status (ARCHITECTURE §5, progress mapping).
PROGRESS_RANGE: dict[JobStatus, tuple[int, int]] = {
    JobStatus.QUEUED: (0, 0),
    JobStatus.TRANSCRIBING: (5, 20),
    JobStatus.SUMMARIZING: (20, 50),
    JobStatus.EXTRACTING_FRAMES: (50, 80),
    JobStatus.PUBLISHING: (80, 99),
    JobStatus.COMPLETED: (100, 100),
    JobStatus.COMPLETED_WITH_WARNINGS: (100, 100),
}


def is_terminal(status: JobStatus) -> bool:
    return status in TERMINAL_STATUSES


def can_transition(from_status: JobStatus, to_status: JobStatus) -> bool:
    return from_status in ALLOWED_FROM[to_status]


def status_of(stage: Stage) -> JobStatus:
    return STAGE_STATUS[stage]


def status_before(stage: Stage) -> JobStatus:
    """Status a job is in just before `stage` starts for the first time."""
    index = STAGE_ORDER.index(stage)
    return JobStatus.QUEUED if index == 0 else STAGE_STATUS[STAGE_ORDER[index - 1]]


def next_stage(stage: Stage) -> Stage | None:
    """The stage after `stage`, or None after PUBLISH (the job then becomes terminal)."""
    index = STAGE_ORDER.index(stage)
    return STAGE_ORDER[index + 1] if index + 1 < len(STAGE_ORDER) else None


def progress_at(status: JobStatus, fraction: float = 0.0) -> int:
    """Progress for `status`, `fraction` (0..1) of the way through its range.

    FAILED / DEAD_LETTERED have no range: callers keep the last progress value.
    """
    low, high = PROGRESS_RANGE[status]
    clamped = min(max(fraction, 0.0), 1.0)
    return round(low + (high - low) * clamped)


ERROR_STATUSES: frozenset[JobStatus] = frozenset({JobStatus.FAILED, JobStatus.DEAD_LETTERED})


async def transition(
    conn: AsyncConnection,
    job_id: UUID,
    to: JobStatus,
    *,
    allowed_from: Iterable[JobStatus] | None = None,
    progress: int | None = None,
    error: ErrorInfo | None = None,
    stage: Stage | None = None,
    attempt: int | None = None,
    detail: Mapping[str, Any] | None = None,
    extra_where: ColumnElement[bool] | None = None,
) -> JobStatus | None:
    """Move a job to `to` and append a TRANSITION event, inside the caller's transaction.

    Conditional: `UPDATE jobs ... WHERE id = :id AND status IN (allowed)`. Returns the status the
    job left, or None when the job is missing or already elsewhere — the caller then treats its
    message as stale (ARCHITECTURE §7). `allowed_from` may only narrow `ALLOWED_FROM[to]`;
    `extra_where` adds a predicate (the sweeper requires `enqueued_at IS NULL`). Entering a new
    status resets `progress` to the start of its range unless `progress` is given; any transition
    clears the waiting state; terminal statuses set `finished_at`; FAILED / DEAD_LETTERED need
    `error`.
    """
    # Local import: common.db imports JobStatus from this module.
    from common.db import job_events, jobs

    allowed = ALLOWED_FROM[to] if allowed_from is None else frozenset(allowed_from)
    if not allowed or not allowed <= ALLOWED_FROM[to]:
        raise ValueError(f"cannot transition to {to} from {sorted(allowed)}")
    if (to in ERROR_STATUSES) != (error is not None):
        raise ValueError("error is required for FAILED / DEAD_LETTERED and only for them")

    # Lock the row so the status we report as `from` is the one we actually replaced.
    current = await conn.scalar(select(jobs.c.status).where(jobs.c.id == job_id).with_for_update())
    if current is None or JobStatus(current) not in allowed:
        return None
    from_status = JobStatus(current)

    values: dict[str, Any] = {
        "status": to.value,
        "updated_at": func.now(),
        "waiting_until": None,
        "waiting_reason": None,
    }
    if progress is not None:
        values["progress"] = progress
    elif to in PROGRESS_RANGE and to != from_status:
        values["progress"] = progress_at(to)
    if is_terminal(to):
        values["finished_at"] = func.now()
    if error is not None:
        values["error"] = error.model_dump(mode="json")

    statement = update(jobs).where(
        jobs.c.id == job_id, jobs.c.status.in_([status.value for status in allowed])
    )
    if extra_where is not None:
        statement = statement.where(extra_where)
    result = await conn.execute(statement.values(**values))
    if result.rowcount != 1:
        return None

    event_detail = dict(detail or {})
    if error is not None:
        event_detail["error"] = error.model_dump(mode="json")
    await conn.execute(
        insert(job_events).values(
            job_id=job_id,
            kind="TRANSITION",
            from_status=from_status.value,
            to_status=to.value,
            stage=stage.value if stage is not None else None,
            attempt=attempt,
            detail=event_detail or None,
        )
    )
    return from_status
