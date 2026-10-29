"""POST /jobs, GET /jobs/{id}, GET /jobs/{id}/results (ARCHITECTURE §10).

POST validates, inserts the row, publishes the first stage message with a publisher confirm and
answers 202 — no pipeline work here (invariant 1). GET handlers only read (invariant 10).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Header, Request, Response
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from common.broker import BrokerPublisher, PublishFailedError
from common.crypto import TokenCipher
from common.db import jobs
from common.errors import ErrorCode, PermanentError
from common.logging import get_logger
from common.models import (
    ErrorBody,
    ErrorInfo,
    JobCreatedResponse,
    JobCreateRequest,
    JobResult,
    JobResultsResponse,
    JobStatusResponse,
    JobWarning,
    StageMessage,
    Trace,
    WaitingInfo,
)
from common.state import JobStatus, Stage, is_terminal, transition
from common.youtube_url import canonical_url, parse_video_id
from gateway.errors import ApiError

router = APIRouter()
log = get_logger()

_NOT_FOUND = {"model": ErrorBody, "description": "Unknown or malformed job id"}


@router.post(
    "/jobs",
    status_code=202,
    response_model=JobCreatedResponse,
    responses={
        400: {"model": ErrorBody, "description": "INVALID_URL or MISSING_NOTION_TOKEN"},
        422: {"model": ErrorBody, "description": "Body does not match the schema"},
        503: {"model": ErrorBody, "description": "BROKER_UNAVAILABLE"},
    },
)
async def create_job(
    body: JobCreateRequest,
    request: Request,
    response: Response,
    idempotency_key: Annotated[str | None, Header(max_length=200)] = None,
) -> JobCreatedResponse:
    try:
        video_id = parse_video_id(body.youtube_url)
    except PermanentError as exc:
        raise ApiError(400, ErrorCode.INVALID_URL, exc.message) from exc
    token = body.notion_token.get_secret_value().strip() if body.notion_token else ""
    if not token:
        raise ApiError(400, ErrorCode.MISSING_NOTION_TOKEN, "notion_token is required")

    state = request.app.state
    engine: AsyncEngine = state.engine
    job_id, status, created = await _insert_job(
        engine, video_id, body.notion_parent_id, idempotency_key
    )
    response.headers["Location"] = f"/jobs/{job_id}"
    if not created:
        log.info("job_create_replayed", job_id=str(job_id))
        return JobCreatedResponse(job_id=job_id, status=status, video_id=video_id)

    cipher: TokenCipher = state.cipher
    publisher: BrokerPublisher = state.publisher
    message = StageMessage(
        job_id=job_id,
        stage=Stage.TRANSCRIBE,
        attempt=1,
        notion_token_enc=cipher.encrypt(token),
        notion_parent_id=body.notion_parent_id,
        trace=Trace(first_enqueued_at=datetime.now(UTC)),
    )
    try:
        await publisher.publish(message)
    except PublishFailedError as exc:
        error = ErrorInfo(code=ErrorCode.ENQUEUE_FAILED, message="could not enqueue the job")
        async with engine.begin() as conn:
            await transition(
                conn, job_id, JobStatus.FAILED, allowed_from={JobStatus.QUEUED}, error=error
            )
        raise ApiError(
            503, ErrorCode.BROKER_UNAVAILABLE, "message broker unavailable, try again"
        ) from exc

    try:
        async with engine.begin() as conn:
            await conn.execute(
                update(jobs)
                .where(jobs.c.id == job_id, jobs.c.enqueued_at.is_(None))
                .values(enqueued_at=func.now(), updated_at=func.now())
            )
    except Exception as exc:
        # The message is confirmed, so the job will run; the worker sets enqueued_at on claim.
        log.warning("enqueued_at_not_saved", job_id=str(job_id), error=type(exc).__name__)

    log.info("job_created", job_id=str(job_id), video_id=video_id)
    return JobCreatedResponse(job_id=job_id, status=JobStatus.QUEUED, video_id=video_id)


async def _insert_job(
    engine: AsyncEngine, video_id: str, notion_parent_id: str | None, idempotency_key: str | None
) -> tuple[UUID, JobStatus, bool]:
    """Insert the job row (outbox: enqueued_at NULL). Same Idempotency-Key → the existing job."""
    async with engine.begin() as conn:
        inserted = await conn.scalar(
            insert(jobs)
            .values(
                id=uuid4(),
                video_id=video_id,
                youtube_url=canonical_url(video_id),
                notion_parent_id=notion_parent_id,
                idempotency_key=idempotency_key,
            )
            .on_conflict_do_nothing(index_elements=[jobs.c.idempotency_key])
            .returning(jobs.c.id)
        )
        if inserted is not None:
            return inserted, JobStatus.QUEUED, True
        existing = (
            await conn.execute(
                select(jobs.c.id, jobs.c.status).where(jobs.c.idempotency_key == idempotency_key)
            )
        ).one()
        return existing.id, JobStatus(existing.status), False


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, responses={404: _NOT_FOUND})
async def get_job(job_id: str, request: Request) -> JobStatusResponse:
    return job_status_response(await _load_job(request.app.state.engine, job_id))


@router.get(
    "/jobs/{job_id}/results",
    response_model=JobResultsResponse,
    responses={404: _NOT_FOUND, 409: {"model": ErrorBody, "description": "JOB_NOT_TERMINAL"}},
)
async def get_results(job_id: str, request: Request) -> JobResultsResponse:
    row = await _load_job(request.app.state.engine, job_id)
    status = JobStatus(row["status"])
    if not is_terminal(status):
        raise ApiError(409, ErrorCode.JOB_NOT_TERMINAL, f"job is {status}; results not ready")
    return JobResultsResponse(
        status=status, result=_result(row), warnings=_warnings(row), error=_error(row)
    )


async def _load_job(engine: AsyncEngine, raw_id: str) -> Mapping[str, Any]:
    try:
        job_id = UUID(raw_id)
    except ValueError:
        job_id = None  # malformed id → 404 like an unknown one (D15)
    if job_id is not None:
        async with engine.connect() as conn:
            row = (await conn.execute(select(jobs).where(jobs.c.id == job_id))).mappings().first()
        if row is not None:
            return dict(row)
    raise ApiError(404, ErrorCode.JOB_NOT_FOUND, "job not found")


def job_status_response(row: Mapping[str, Any]) -> JobStatusResponse:
    status = JobStatus(row["status"])
    waiting = None
    if row["waiting_until"] is not None and not is_terminal(status):
        waiting = WaitingInfo(until=row["waiting_until"], reason=row["waiting_reason"] or "")
    return JobStatusResponse(
        job_id=row["id"],
        status=status,
        progress=row["progress"],
        video_title=row["video_title"],
        duration_sec=row["duration_sec"],
        warnings=_warnings(row),
        error=_error(row),
        result=_result(row),
        waiting=waiting,
    )


def _warnings(row: Mapping[str, Any]) -> list[JobWarning]:
    return [JobWarning.model_validate(item) for item in row["warnings"] or []]


def _error(row: Mapping[str, Any]) -> ErrorInfo | None:
    return ErrorInfo.model_validate(row["error"]) if row["error"] else None


def _result(row: Mapping[str, Any]) -> JobResult | None:
    if row["status"] not in (JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS):
        return None
    if not row["notion_page_url"]:
        return None
    return JobResult(
        notion_page_url=row["notion_page_url"],
        frames_requested=row["frames_requested"] or 0,
        frames_published=row["frames_published"] or 0,
    )
