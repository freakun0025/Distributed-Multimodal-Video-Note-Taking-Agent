"""consume -> checkpoint -> publish next -> ack; retry/DLQ routing on errors (ARCHITECTURE §7).

Invariant 3: a message is acked only after its stage output is durable (artefact + checkpoint
row) and the next message (or retry / DLQ copy) is publisher-confirmed. Anything that fails
before that point leaves the message unacked, so RabbitMQ redelivers it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import ValidationError

from common.broker import DLQ_ROUTING_KEY, WORK_ROUTING_KEY, retry_delay, retry_routing_key
from common.config import Settings
from common.errors import ErrorCode, PermanentError, RetryableError
from common.logging import bind_job, clear_job, get_logger
from common.models import ErrorInfo, StageMessage
from common.state import (
    STAGE_ORDER,
    STAGE_STATUS,
    JobStatus,
    Stage,
    is_terminal,
    next_stage,
)
from worker.context import JobRecord, StageContext, StageFn
from worker.store import JobStore

log = get_logger()

# x-delivery-count: set by quorum queues on messages redelivered after a consumer crash,
# channel close or reject — not after a nack (RabbitMQ 4.3, ARCHITECTURE §6.1).
DELIVERY_COUNT_HEADER = "x-delivery-count"


class IncomingMessage(Protocol):
    """The parts of aio_pika.abc.AbstractIncomingMessage the runtime uses."""

    @property
    def body(self) -> bytes: ...

    @property
    def headers(self) -> Mapping[str, Any]: ...

    async def ack(self, multiple: bool = False) -> None: ...

    async def reject(self, requeue: bool = False) -> None: ...

    async def nack(self, multiple: bool = False, requeue: bool = True) -> None: ...


class Publisher(Protocol):
    async def publish(self, message: StageMessage, routing_key: str = WORK_ROUTING_KEY) -> None: ...


class Runtime:
    def __init__(
        self,
        store: JobStore,
        publisher: Publisher,
        settings: Settings,
        stages: Mapping[Stage, StageFn],
        worker: str,
    ) -> None:
        self._store = store
        self._publisher = publisher
        self._settings = settings
        self._stages = stages
        self._worker = worker

    async def handle(self, incoming: IncomingMessage) -> None:
        try:
            message = StageMessage.from_body(incoming.body)
        except ValidationError:
            # Unparseable: reject without requeue, so jobs.work dead-letters it to jobs.dlq.
            log.error("message_malformed", size=len(incoming.body))
            await incoming.reject(requeue=False)
            return
        bind_job(str(message.job_id), stage=message.stage.value, attempt=message.attempt)
        try:
            await self._process(message, incoming)
        except Exception as exc:
            # The store or broker failed outside the stage (e.g. Postgres down): retryable.
            log.error("message_handling_failed", error=type(exc).__name__)
            await self._route_or_requeue(
                message,
                incoming,
                RetryableError(ErrorCode.INTERNAL_ERROR, f"worker error: {type(exc).__name__}"),
            )
        finally:
            clear_job()

    async def _process(self, message: StageMessage, incoming: IncomingMessage) -> None:
        job = await self._store.load(message.job_id)
        if job is None:
            log.warning("message_dropped", reason="job_missing")
            await incoming.ack()
            return
        if is_terminal(job.status):
            log.info("message_dropped", reason="job_terminal", status=job.status)
            await incoming.ack()
            return

        delivery_count = _delivery_count(incoming.headers)
        if delivery_count:
            log.warning("message_redelivered", delivery_count=delivery_count)
            await self._store.record_redelivery(
                job.id, message.stage, message.attempt, delivery_count, self._worker
            )

        if await self._store.has_checkpoint(job.id, message.stage):
            # Crashed after the checkpoint, before (or after) the next publish: only advance.
            log.info("stage_already_checkpointed")
            if not _already_beyond(job.status, message.stage):
                await self._publish_next(message)
            await incoming.ack()
            return

        if delivery_count >= self._settings.retry_max_attempts:
            # This stage keeps killing workers: stop before RabbitMQ's own delivery limit.
            await self._dead_letter(
                message,
                ErrorInfo(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"worker crashed {delivery_count} times running {message.stage}",
                ),
            )
            await incoming.ack()
            return

        if not await self._store.start_stage(job.id, message.stage, message.attempt, self._worker):
            log.info("message_dropped", reason="stale", status=job.status)
            await incoming.ack()
            return

        await self._run_stage(job, message, incoming)

    async def _run_stage(
        self, job: JobRecord, message: StageMessage, incoming: IncomingMessage
    ) -> None:
        stage = message.stage
        ctx = StageContext(
            job=job,
            message=message,
            job_dir=self._settings.jobs_dir / str(job.id),
            settings=self._settings,
            log=log,
            report_progress=lambda fraction: self._store.report_progress(job.id, stage, fraction),
        )
        log.info("stage_started")
        try:
            outcome = await self._stages[stage](ctx)
        except RetryableError as exc:
            await self._route_failure(message, incoming, exc)
            return
        except PermanentError as exc:
            log.warning("stage_failed", code=exc.code)
            await self._store.finish_with_error(
                job.id,
                JobStatus.FAILED,
                stage,
                message.attempt,
                ErrorInfo.model_validate(exc.to_info()),
                self._worker,
            )
            await incoming.ack()
            return
        except Exception as exc:  # unknown → retryable; the attempt budget bounds it
            log.exception("stage_crashed", error=type(exc).__name__)
            await self._route_failure(
                message,
                incoming,
                RetryableError(ErrorCode.INTERNAL_ERROR, f"{type(exc).__name__} in {stage}"),
            )
            return

        first = await self._store.complete_stage(
            job.id, stage, message.attempt, outcome, self._worker
        )
        if first:
            await self._publish_next(message)
            log.info("stage_completed")
        else:
            # Another delivery checkpointed first; it (or its redelivery) publishes the next one.
            log.info("stage_completed_elsewhere")
        await incoming.ack()

    async def _publish_next(self, message: StageMessage) -> None:
        following = next_stage(message.stage)
        if following is not None:
            await self._publisher.publish(message.for_stage(following), WORK_ROUTING_KEY)

    async def _route_failure(
        self, message: StageMessage, incoming: IncomingMessage, error: RetryableError
    ) -> None:
        """Retry through a delay queue while the budget lasts, else DLQ + DEAD_LETTERED; ack."""
        if message.attempt < self._settings.retry_max_attempts:
            delay = retry_delay(self._settings.retry_delays, message.attempt, error.retry_after_s)
            await self._publisher.publish(message.for_retry(), retry_routing_key(delay))
            log.warning("retry_scheduled", code=error.code, delay_s=delay)
            try:
                await self._store.record_retry(
                    message.job_id, message.stage, message.attempt, error, delay, self._worker
                )
            except Exception as exc:  # display-only state; the retry itself is confirmed
                log.warning("retry_not_recorded", error=type(exc).__name__)
        else:
            await self._dead_letter(message, ErrorInfo.model_validate(error.to_info()))
        await incoming.ack()

    async def _dead_letter(self, message: StageMessage, error: ErrorInfo) -> None:
        await self._publisher.publish(message, DLQ_ROUTING_KEY)
        log.error("dead_lettered", code=error.code)
        try:
            await self._store.finish_with_error(
                message.job_id,
                JobStatus.DEAD_LETTERED,
                message.stage,
                message.attempt,
                error,
                self._worker,
            )
        except Exception as exc:
            # The copy is safely in jobs.dlq; the row stays non-terminal (ARCHITECTURE §15).
            log.error("dead_letter_not_recorded", error=type(exc).__name__)

    async def _route_or_requeue(
        self, message: StageMessage, incoming: IncomingMessage, error: RetryableError
    ) -> None:
        try:
            await self._route_failure(message, incoming, error)
        except Exception as exc:
            # Broker trouble too: hand the message back; RabbitMQ redelivers it.
            log.error("message_requeued", error=type(exc).__name__)
            try:
                await incoming.nack(requeue=True)
            except Exception:
                log.error("message_requeue_failed")  # channel gone: redelivered on reconnect


def _delivery_count(headers: Mapping[str, Any] | None) -> int:
    value = (headers or {}).get(DELIVERY_COUNT_HEADER)
    try:
        return max(int(value), 0) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _already_beyond(status: JobStatus, stage: Stage) -> bool:
    """True when the job is already in a later stage, so the next message was consumed."""
    later = STAGE_ORDER[STAGE_ORDER.index(stage) + 1 :]
    return status in {STAGE_STATUS[s] for s in later}
