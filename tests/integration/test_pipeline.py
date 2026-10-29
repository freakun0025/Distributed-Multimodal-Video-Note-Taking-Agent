"""Jobs through the real gateway, broker, Postgres and >= 2 workers running M1 stub stages.

The kill test SIGKILLs a worker container with the docker CLI, so it needs `docker` on PATH.
"""

import asyncio
import shutil
import time
from typing import Any
from uuid import UUID

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from common.db import job_events, jobs, stage_checkpoints
from common.state import STAGE_ORDER, TERMINAL_STATUSES
from tests.integration.conftest import VIDEO_URL, JobRows

pytestmark = pytest.mark.integration

BODY = {"youtube_url": VIDEO_URL, "notion_token": "integration-test-token"}
# Four stub stages of STUB_STAGE_SECONDS each, plus queueing behind other jobs.
JOB_TIMEOUT_S = 90


async def _submit(gateway: httpx.AsyncClient, job_rows: JobRows) -> UUID:
    response = await gateway.post("/jobs", json=BODY)
    assert response.status_code == 202, response.text
    return job_rows.track(response.json()["job_id"])


async def _wait_terminal(gateway: httpx.AsyncClient, job_id: UUID) -> dict[str, Any]:
    deadline = time.monotonic() + JOB_TIMEOUT_S
    while time.monotonic() < deadline:
        body: dict[str, Any] = (await gateway.get(f"/jobs/{job_id}")).json()
        if body["status"] in TERMINAL_STATUSES:
            return body
        await asyncio.sleep(0.5)
    raise AssertionError(f"job {job_id} not terminal after {JOB_TIMEOUT_S}s: {body}")


async def _events(engine: AsyncEngine, job_id: UUID) -> list[dict[str, Any]]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            select(job_events).where(job_events.c.job_id == job_id).order_by(job_events.c.id)
        )
        return [dict(row) for row in rows.mappings()]


async def _checkpoints(engine: AsyncEngine, job_id: UUID) -> list[str]:
    async with engine.connect() as conn:
        rows = await conn.scalars(
            select(stage_checkpoints.c.stage).where(stage_checkpoints.c.job_id == job_id)
        )
        return sorted(rows.all())


def _transitions(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(e["from_status"], e["to_status"]) for e in events if e["kind"] == "TRANSITION"]


async def test_job_reaches_completed_through_every_stage(
    gateway: httpx.AsyncClient, engine: AsyncEngine, job_rows: JobRows
) -> None:
    job_id = await _submit(gateway, job_rows)
    final = await _wait_terminal(gateway, job_id)

    assert final["status"] == "COMPLETED"
    assert final["progress"] == 100
    results = await gateway.get(f"/jobs/{job_id}/results")
    assert results.status_code == 200
    assert results.json()["status"] == "COMPLETED"
    assert _transitions(await _events(engine, job_id)) == [
        ("QUEUED", "TRANSCRIBING"),
        ("TRANSCRIBING", "SUMMARIZING"),
        ("SUMMARIZING", "EXTRACTING_FRAMES"),
        ("EXTRACTING_FRAMES", "PUBLISHING"),
        ("PUBLISHING", "COMPLETED"),
    ]
    assert await _checkpoints(engine, job_id) == sorted(stage.value for stage in STAGE_ORDER)
    async with engine.connect() as conn:
        row = (await conn.execute(select(jobs).where(jobs.c.id == job_id))).mappings().one()
    assert row["enqueued_at"] is not None
    assert row["finished_at"] is not None


async def test_concurrent_jobs_are_shared_between_workers(
    gateway: httpx.AsyncClient, engine: AsyncEngine, job_rows: JobRows
) -> None:
    job_ids = [await _submit(gateway, job_rows) for _ in range(4)]
    finals = await asyncio.gather(*(_wait_terminal(gateway, job_id) for job_id in job_ids))
    assert {final["status"] for final in finals} == {"COMPLETED"}
    workers = {
        event["detail"]["worker"]
        for job_id in job_ids
        for event in await _events(engine, job_id)
        if event["kind"] == "TRANSITION"
    }
    assert len(workers) >= 2  # Spec §1.1: at least two workers do the work


async def _docker(*args: str) -> None:
    process = await asyncio.create_subprocess_exec("docker", *args)
    assert await process.wait() == 0, f"docker {' '.join(args)} failed"


@pytest.mark.skipif(shutil.which("docker") is None, reason="needs the docker CLI")
async def test_killing_the_worker_mid_stage_still_completes_the_job_once(
    gateway: httpx.AsyncClient, engine: AsyncEngine, job_rows: JobRows
) -> None:
    job_id = await _submit(gateway, job_rows)

    # Wait until a worker has entered EXTRACT_FRAMES, then SIGKILL that worker's container.
    victim = None
    deadline = time.monotonic() + JOB_TIMEOUT_S
    while victim is None and time.monotonic() < deadline:
        for event in await _events(engine, job_id):
            if event["kind"] == "TRANSITION" and event["to_status"] == "EXTRACTING_FRAMES":
                victim = event["detail"]["worker"]  # container hostname = short container id
        await asyncio.sleep(0.1)
    assert victim is not None, "job never reached EXTRACTING_FRAMES"
    await _docker("kill", "--signal", "KILL", victim)
    try:
        final = await _wait_terminal(gateway, job_id)
    finally:
        await _docker("start", victim)

    assert final["status"] == "COMPLETED"
    events = await _events(engine, job_id)
    frames = [e for e in events if e["stage"] == "EXTRACT_FRAMES"]
    redelivered = [e for e in frames if e["kind"] == "REDELIVERED"]
    assert redelivered, "the killed worker's message was not redelivered"
    assert redelivered[0]["detail"]["worker"] != victim
    assert redelivered[0]["detail"]["delivery_count"] >= 1
    # Each stage checkpointed exactly once, and the job finished exactly once.
    assert await _checkpoints(engine, job_id) == sorted(stage.value for stage in STAGE_ORDER)
    assert [t for t in _transitions(events) if t[1] == "COMPLETED"] == [("PUBLISHING", "COMPLETED")]
