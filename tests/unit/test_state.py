from typing import Any
from uuid import uuid4

import pytest

from common.errors import ErrorCode
from common.models import ErrorInfo
from common.state import (
    ALLOWED_FROM,
    NON_TERMINAL_STATUSES,
    STAGE_ORDER,
    TERMINAL_STATUSES,
    JobStatus,
    Stage,
    can_transition,
    is_terminal,
    next_stage,
    progress_at,
    status_before,
    status_of,
    transition,
)

S = JobStatus


def test_status_strings_match_the_spec_exactly() -> None:
    assert [s.value for s in JobStatus] == [
        "QUEUED",
        "TRANSCRIBING",
        "SUMMARIZING",
        "EXTRACTING_FRAMES",
        "PUBLISHING",
        "COMPLETED",
        "COMPLETED_WITH_WARNINGS",
        "FAILED",
        "DEAD_LETTERED",
    ]


def test_every_status_has_an_entry_in_the_transition_table() -> None:
    assert set(ALLOWED_FROM) == set(JobStatus)


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        (S.QUEUED, S.TRANSCRIBING),
        (S.TRANSCRIBING, S.SUMMARIZING),
        (S.SUMMARIZING, S.EXTRACTING_FRAMES),
        (S.EXTRACTING_FRAMES, S.PUBLISHING),
        (S.PUBLISHING, S.COMPLETED),
        (S.PUBLISHING, S.COMPLETED_WITH_WARNINGS),
    ],
)
def test_the_spec_happy_path_transitions_are_allowed(
    from_status: JobStatus, to_status: JobStatus
) -> None:
    assert can_transition(from_status, to_status)


@pytest.mark.parametrize(
    "status", [S.TRANSCRIBING, S.SUMMARIZING, S.EXTRACTING_FRAMES, S.PUBLISHING]
)
def test_redelivery_may_reenter_the_same_stage_status(status: JobStatus) -> None:
    assert can_transition(status, status)


@pytest.mark.parametrize("from_status", sorted(NON_TERMINAL_STATUSES))
@pytest.mark.parametrize("to_status", [S.FAILED, S.DEAD_LETTERED])
def test_any_non_terminal_status_can_fail_or_dead_letter(
    from_status: JobStatus, to_status: JobStatus
) -> None:
    assert can_transition(from_status, to_status)


@pytest.mark.parametrize("from_status", sorted(TERMINAL_STATUSES))
@pytest.mark.parametrize("to_status", list(JobStatus))
def test_terminal_statuses_never_move_again(from_status: JobStatus, to_status: JobStatus) -> None:
    assert not can_transition(from_status, to_status)


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        (S.QUEUED, S.SUMMARIZING),  # skipping a stage
        (S.TRANSCRIBING, S.PUBLISHING),
        (S.SUMMARIZING, S.TRANSCRIBING),  # going backwards
        (S.EXTRACTING_FRAMES, S.COMPLETED),  # finishing without publishing
        (S.QUEUED, S.QUEUED),  # QUEUED is only inserted
    ],
)
def test_out_of_order_transitions_are_rejected(
    from_status: JobStatus, to_status: JobStatus
) -> None:
    assert not can_transition(from_status, to_status)


def test_terminal_set() -> None:
    assert is_terminal(S.COMPLETED)
    assert is_terminal(S.DEAD_LETTERED)
    assert not is_terminal(S.PUBLISHING)


def test_stage_chain_and_status_mapping() -> None:
    assert STAGE_ORDER == (Stage.TRANSCRIBE, Stage.SUMMARIZE, Stage.EXTRACT_FRAMES, Stage.PUBLISH)
    assert [status_of(stage) for stage in STAGE_ORDER] == [
        S.TRANSCRIBING,
        S.SUMMARIZING,
        S.EXTRACTING_FRAMES,
        S.PUBLISHING,
    ]
    assert status_before(Stage.TRANSCRIBE) is S.QUEUED
    assert status_before(Stage.PUBLISH) is S.EXTRACTING_FRAMES
    assert next_stage(Stage.TRANSCRIBE) is Stage.SUMMARIZE
    assert next_stage(Stage.PUBLISH) is None


def test_every_stage_can_be_entered_from_its_predecessor() -> None:
    for stage in STAGE_ORDER:
        assert can_transition(status_before(stage), status_of(stage))


def test_progress_follows_the_documented_ranges() -> None:
    assert progress_at(S.QUEUED) == 0
    assert progress_at(S.SUMMARIZING) == 20
    assert progress_at(S.SUMMARIZING, 1.0) == 50
    assert progress_at(S.EXTRACTING_FRAMES, 0.5) == 65
    assert progress_at(S.PUBLISHING, 2.0) == 99  # clamped
    assert progress_at(S.COMPLETED) == 100


# transition() rejects bad arguments before touching the database; its SQL is covered by
# tests/integration/test_transition.py.
_NO_DB: Any = None


async def test_transition_refuses_to_enter_queued() -> None:
    with pytest.raises(ValueError):
        await transition(_NO_DB, uuid4(), JobStatus.QUEUED)


async def test_transition_allowed_from_may_only_narrow_the_table() -> None:
    with pytest.raises(ValueError):
        await transition(_NO_DB, uuid4(), JobStatus.SUMMARIZING, allowed_from={JobStatus.QUEUED})


async def test_transition_to_failed_requires_an_error_and_others_reject_one() -> None:
    error = ErrorInfo(code=ErrorCode.INTERNAL_ERROR, message="x")
    with pytest.raises(ValueError):
        await transition(_NO_DB, uuid4(), JobStatus.FAILED)
    with pytest.raises(ValueError):
        await transition(_NO_DB, uuid4(), JobStatus.TRANSCRIBING, error=error)
