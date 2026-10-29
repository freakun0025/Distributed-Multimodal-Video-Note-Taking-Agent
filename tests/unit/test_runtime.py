"""Runtime decisions (ARCHITECTURE §7) against in-memory fakes; SQL is in tests/integration.

Every fake appends to one shared `events` list, so tests assert the order of side effects —
in particular that `ack` comes only after the confirmed publish (invariant 3).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from common.broker import PublishFailedError
from common.config import Settings
from common.errors import ErrorCode, ExternalApi, PermanentError, RetryableError
from common.models import ErrorInfo, StageMessage
from common.state import JobStatus, Stage
from worker.context import JobRecord, StageContext, StageFn, StageOutcome
from worker.runtime import Runtime

Event = tuple[Any, ...]
JOB_ID = uuid4()


def _job(status: JobStatus = JobStatus.QUEUED) -> JobRecord:
    return JobRecord(
        id=JOB_ID,
        video_id="dQw4w9WgXcQ",
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        status=status,
        progress=0,
        video_title=None,
        duration_sec=None,
        notion_parent_id=None,
        notion_page_id=None,
        notion_page_url=None,
        warnings=(),
        enqueued_at=None,
    )


def _message(stage: Stage = Stage.TRANSCRIBE, attempt: int = 1) -> StageMessage:
    return StageMessage(job_id=JOB_ID, stage=stage, attempt=attempt, notion_token_enc="enc")


class FakeStore:
    def __init__(
        self,
        events: list[Event],
        job: JobRecord | None,
        *,
        checkpoints: frozenset[Stage] = frozenset(),
        start_ok: bool = True,
        first_to_checkpoint: bool = True,
        load_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.job = job
        self.checkpoints = checkpoints
        self.start_ok = start_ok
        self.first_to_checkpoint = first_to_checkpoint
        self.load_error = load_error

    async def load(self, job_id: UUID) -> JobRecord | None:
        if self.load_error is not None:
            raise self.load_error
        return self.job

    async def has_checkpoint(self, job_id: UUID, stage: Stage) -> bool:
        return stage in self.checkpoints

    async def start_stage(self, job_id: UUID, stage: Stage, attempt: int, worker: str) -> bool:
        self.events.append(("start", stage))
        return self.start_ok

    async def report_progress(self, job_id: UUID, stage: Stage, fraction: float) -> None:
        self.events.append(("progress", stage, fraction))

    async def complete_stage(
        self, job_id: UUID, stage: Stage, attempt: int, outcome: StageOutcome, worker: str
    ) -> bool:
        self.events.append(("checkpoint", stage, dict(outcome.summary)))
        return self.first_to_checkpoint

    async def record_retry(
        self,
        job_id: UUID,
        stage: Stage,
        attempt: int,
        error: RetryableError,
        delay_s: int,
        worker: str,
    ) -> None:
        self.events.append(("waiting", error.waiting_reason, delay_s))

    async def finish_with_error(
        self,
        job_id: UUID,
        status: JobStatus,
        stage: Stage,
        attempt: int,
        error: ErrorInfo,
        worker: str,
    ) -> bool:
        self.events.append(("finish", status, error.code))
        return True

    async def record_redelivery(
        self, job_id: UUID, stage: Stage, attempt: int, delivery_count: int, worker: str
    ) -> None:
        self.events.append(("redelivered", delivery_count))


class FakePublisher:
    def __init__(self, events: list[Event], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail

    async def publish(self, message: StageMessage, routing_key: str = "work") -> None:
        if self.fail:
            raise PublishFailedError("no confirm")
        self.events.append(("publish", routing_key, message.stage, message.attempt))


class FakeIncoming:
    def __init__(
        self, events: list[Event], body: bytes, headers: Mapping[str, Any] | None = None
    ) -> None:
        self.events = events
        self.body = body
        self.headers = dict(headers or {})

    async def ack(self, multiple: bool = False) -> None:
        self.events.append(("ack",))

    async def reject(self, requeue: bool = False) -> None:
        self.events.append(("reject", requeue))

    async def nack(self, multiple: bool = False, requeue: bool = True) -> None:
        self.events.append(("nack", requeue))


def _stage_that(events: list[Event], effect: Exception | None = None) -> StageFn:
    async def run(ctx: StageContext) -> StageOutcome:
        events.append(("run", ctx.message.stage))
        await ctx.report_progress(0.5)
        if effect is not None:
            raise effect
        return StageOutcome(summary={"ok": True})

    return run


async def _handle(
    *,
    tmp_path: Path,
    job: JobRecord | None = None,
    message: StageMessage | None = None,
    body: bytes | None = None,
    headers: Mapping[str, Any] | None = None,
    stage_error: Exception | None = None,
    publish_fails: bool = False,
    **store_options: Any,
) -> list[Event]:
    events: list[Event] = []
    store = FakeStore(events, job if job is not None else _job(), **store_options)
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, retry_max_attempts=3, retry_delays=[10, 30, 90], data_dir=tmp_path
    )
    stage_fn = _stage_that(events, stage_error)
    runtime = Runtime(
        store,
        FakePublisher(events, fail=publish_fails),
        settings,
        dict.fromkeys(Stage, stage_fn),
        worker="worker-a",
    )
    raw = body if body is not None else (message or _message()).to_body()
    await runtime.handle(FakeIncoming(events, raw, headers))
    return events


async def test_completed_stage_publishes_the_next_stage_and_only_then_acks(
    tmp_path: Path,
) -> None:
    events = await _handle(tmp_path=tmp_path)
    assert events == [
        ("start", Stage.TRANSCRIBE),
        ("run", Stage.TRANSCRIBE),
        ("progress", Stage.TRANSCRIBE, 0.5),
        ("checkpoint", Stage.TRANSCRIBE, {"ok": True}),
        ("publish", "work", Stage.SUMMARIZE, 1),
        ("ack",),
    ]


async def test_the_last_stage_finishes_without_publishing(tmp_path: Path) -> None:
    events = await _handle(
        tmp_path=tmp_path, job=_job(JobStatus.EXTRACTING_FRAMES), message=_message(Stage.PUBLISH)
    )
    assert [e[0] for e in events] == ["start", "run", "progress", "checkpoint", "ack"]


@pytest.mark.parametrize("status", [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.DEAD_LETTERED])
async def test_message_for_a_terminal_job_is_acked_without_running(
    tmp_path: Path, status: JobStatus
) -> None:
    assert await _handle(tmp_path=tmp_path, job=_job(status)) == [("ack",)]


async def test_message_for_a_missing_job_is_acked_and_dropped(tmp_path: Path) -> None:
    events: list[Event] = []
    store = FakeStore(events, None)
    settings = Settings(_env_file=None, data_dir=tmp_path)  # type: ignore[call-arg]
    runtime = Runtime(store, FakePublisher(events), settings, {}, worker="w")
    await runtime.handle(FakeIncoming(events, _message().to_body()))
    assert events == [("ack",)]


async def test_redelivery_after_checkpoint_does_not_rerun_stage(tmp_path: Path) -> None:
    events = await _handle(
        tmp_path=tmp_path,
        job=_job(JobStatus.TRANSCRIBING),
        checkpoints=frozenset({Stage.TRANSCRIBE}),
        headers={"x-delivery-count": 1},
    )
    assert events == [("redelivered", 1), ("publish", "work", Stage.SUMMARIZE, 1), ("ack",)]


async def test_checkpointed_stage_is_not_republished_once_the_job_has_moved_on(
    tmp_path: Path,
) -> None:
    events = await _handle(
        tmp_path=tmp_path,
        job=_job(JobStatus.EXTRACTING_FRAMES),
        message=_message(Stage.SUMMARIZE),
        checkpoints=frozenset({Stage.SUMMARIZE}),
    )
    assert events == [("ack",)]


async def test_stale_message_is_acked_without_running(tmp_path: Path) -> None:
    events = await _handle(tmp_path=tmp_path, start_ok=False)
    assert events == [("start", Stage.TRANSCRIBE), ("ack",)]


async def test_losing_the_checkpoint_race_acks_without_publishing(tmp_path: Path) -> None:
    events = await _handle(tmp_path=tmp_path, first_to_checkpoint=False)
    assert [e[0] for e in events] == ["start", "run", "progress", "checkpoint", "ack"]


async def test_retryable_error_goes_to_the_delay_queue_with_the_next_attempt(
    tmp_path: Path,
) -> None:
    error = RetryableError(ErrorCode.UPSTREAM_ERROR, "5xx")
    events = await _handle(tmp_path=tmp_path, stage_error=error)
    assert events[-3:] == [
        ("publish", "retry.10s", Stage.TRANSCRIBE, 2),
        ("waiting", "UPSTREAM_ERROR", 10),
        ("ack",),
    ]


async def test_second_retry_uses_the_second_delay(tmp_path: Path) -> None:
    error = RetryableError(ErrorCode.UPSTREAM_TIMEOUT, "slow")
    events = await _handle(tmp_path=tmp_path, message=_message(attempt=2), stage_error=error)
    assert ("publish", "retry.30s", Stage.TRANSCRIBE, 3) in events


async def test_retry_after_rounds_up_to_the_next_delay_queue(tmp_path: Path) -> None:
    error = RetryableError(ErrorCode.RATE_LIMITED, "429", api=ExternalApi.LLM, retry_after_s=45)
    events = await _handle(tmp_path=tmp_path, stage_error=error)
    assert ("publish", "retry.90s", Stage.TRANSCRIBE, 2) in events
    assert ("waiting", "RATE_LIMITED:llm", 90) in events


async def test_exhausted_budget_dead_letters_the_job(tmp_path: Path) -> None:
    error = RetryableError(ErrorCode.VIDEO_UNAVAILABLE, "private")
    events = await _handle(tmp_path=tmp_path, message=_message(attempt=3), stage_error=error)
    assert events[-3:] == [
        ("publish", "dlq", Stage.TRANSCRIBE, 3),
        ("finish", JobStatus.DEAD_LETTERED, ErrorCode.VIDEO_UNAVAILABLE),
        ("ack",),
    ]


async def test_permanent_error_fails_the_job_without_a_retry(tmp_path: Path) -> None:
    error = PermanentError(ErrorCode.VIDEO_TOO_LONG, "25 min")
    events = await _handle(tmp_path=tmp_path, stage_error=error)
    assert events[-2:] == [("finish", JobStatus.FAILED, ErrorCode.VIDEO_TOO_LONG), ("ack",)]
    assert not [e for e in events if e[0] == "publish"]


async def test_unexpected_exception_is_retried_as_internal_error(tmp_path: Path) -> None:
    events = await _handle(tmp_path=tmp_path, stage_error=KeyError("bug"))
    assert ("publish", "retry.10s", Stage.TRANSCRIBE, 2) in events
    assert ("waiting", "INTERNAL_ERROR", 10) in events
    assert events[-1] == ("ack",)


async def test_unconfirmed_next_publish_leaves_the_message_unacked(tmp_path: Path) -> None:
    events = await _handle(tmp_path=tmp_path, publish_fails=True)
    assert ("ack",) not in events
    assert events[-1] == ("nack", True)


async def test_a_stage_that_keeps_crashing_workers_is_dead_lettered(tmp_path: Path) -> None:
    events = await _handle(
        tmp_path=tmp_path, job=_job(JobStatus.TRANSCRIBING), headers={"x-delivery-count": 3}
    )
    assert events == [
        ("redelivered", 3),
        ("publish", "dlq", Stage.TRANSCRIBE, 1),
        ("finish", JobStatus.DEAD_LETTERED, ErrorCode.INTERNAL_ERROR),
        ("ack",),
    ]


async def test_a_crash_redelivery_within_budget_reruns_the_stage(tmp_path: Path) -> None:
    events = await _handle(
        tmp_path=tmp_path, job=_job(JobStatus.TRANSCRIBING), headers={"x-delivery-count": 1}
    )
    assert events[0] == ("redelivered", 1)
    assert ("run", Stage.TRANSCRIBE) in events
    assert events[-2:] == [("publish", "work", Stage.SUMMARIZE, 1), ("ack",)]


async def test_malformed_body_is_rejected_so_it_dead_letters(tmp_path: Path) -> None:
    assert await _handle(tmp_path=tmp_path, body=b"not json") == [("reject", False)]


async def test_store_outage_is_routed_as_a_retry(tmp_path: Path) -> None:
    events = await _handle(tmp_path=tmp_path, load_error=ConnectionRefusedError())
    assert events == [
        ("publish", "retry.10s", Stage.TRANSCRIBE, 2),
        ("waiting", "INTERNAL_ERROR", 10),
        ("ack",),
    ]
