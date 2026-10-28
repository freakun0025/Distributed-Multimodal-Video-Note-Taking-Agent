"""Job statuses, pipeline stages and the transition table (ARCHITECTURE §5, §6.3, §7).

`transition()` — the only writer of `jobs.status` — is added in M1 on top of this table.
"""

from __future__ import annotations

from enum import StrEnum


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
