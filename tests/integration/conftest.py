"""Integration fixtures: talk to the running compose stack from the host (`make up` first).

Override the endpoints with ITEST_GATEWAY_URL, ITEST_DATABASE_URL and ITEST_RABBITMQ_API.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import delete, insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from common.db import create_engine, jobs
from common.state import STAGE_ORDER, JobStatus, status_of, transition

GATEWAY_URL = os.environ.get("ITEST_GATEWAY_URL", "http://localhost:8000")
DATABASE_URL = os.environ.get(
    "ITEST_DATABASE_URL", "postgresql+asyncpg://vna:vna@localhost:5433/vna"
)
RABBITMQ_API = os.environ.get("ITEST_RABBITMQ_API", "http://localhost:15672/api")
RABBITMQ_AUTH = ("vna", "vna")

VIDEO_ID = "dQw4w9WgXcQ"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    db = create_engine(DATABASE_URL)
    yield db
    await db.dispose()


class JobRows:
    """Creates job rows for a test and deletes them (and their events) afterwards."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self.ids: list[UUID] = []

    def track(self, job_id: UUID | str) -> UUID:
        self.ids.append(UUID(str(job_id)))
        return self.ids[-1]

    async def create(self, **columns: object) -> UUID:
        """Insert a QUEUED row; other statuses are reached with `advance` (never written here)."""
        job_id = self.track(uuid4())
        async with self.engine.begin() as conn:
            await conn.execute(
                insert(jobs).values(
                    id=job_id,
                    video_id=VIDEO_ID,
                    youtube_url=VIDEO_URL,
                    **columns,
                )
            )
        return job_id

    async def cleanup(self) -> None:
        if self.ids:
            async with self.engine.begin() as conn:
                await conn.execute(delete(jobs).where(jobs.c.id.in_(self.ids)))


@pytest.fixture
async def job_rows(engine: AsyncEngine) -> AsyncIterator[JobRows]:
    rows = JobRows(engine)
    yield rows
    await rows.cleanup()


async def advance(conn: AsyncConnection, job_id: UUID, to: JobStatus) -> None:
    """Walk a QUEUED job through the stage statuses up to `to`, using transition() only."""
    for stage in STAGE_ORDER:
        target = status_of(stage)
        moved = await transition(conn, job_id, target, stage=stage, attempt=1)
        assert moved is not None, f"could not enter {target}"
        if target == to:
            return
    raise AssertionError(f"{to} is not a stage status")


@pytest.fixture
async def gateway() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=GATEWAY_URL, timeout=10) as client:
        yield client


@pytest.fixture
async def rabbitmq_api() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=RABBITMQ_API, auth=RABBITMQ_AUTH, timeout=10) as client:
        yield client
