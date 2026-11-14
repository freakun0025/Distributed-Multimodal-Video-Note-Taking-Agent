"""Jobs through the real gateway, broker, Postgres and >= 2 workers. Every stage runs for real
against the fakes service (FAKE_APIS=true): YouTube, the LLM and Notion (PUBLISH, CP8).

The kill test SIGKILLs a worker container with the docker CLI, so it needs `docker` on PATH; it
uses the same helpers as `make kill-worker` (chaos/kill_worker.py).
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

from chaos.kill_worker import docker, follow, kill, wait_running
from common.db import job_events, jobs, stage_checkpoints
from common.state import STAGE_ORDER, TERMINAL_STATUSES, JobStatus, Stage
from tests.integration.conftest import FAKES_URL, VIDEO_URL, JobRows

pytestmark = pytest.mark.integration

BODY = {"youtube_url": VIDEO_URL, "notion_token": "integration-test-token"}
# Fake-backed stages, queueing behind other jobs, and the LLM limiter spacing calls at LLM_RPM
# (a waiting job may sit in a 30 s delay queue).
JOB_TIMEOUT_S = 180


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


async def _notion_pages(job_id: UUID) -> list[dict[str, Any]]:
    """The fake Notion's pages for this job (title ends with the job tag)."""
    async with httpx.AsyncClient(base_url=FAKES_URL, timeout=5) as fakes:
        pages: list[dict[str, Any]] = (await fakes.get("/_control/notion/pages")).json()
    return [page for page in pages if page["title"].endswith(f" · #{str(job_id)[:8]}")]


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
    # Real frames from the fake video (ffmpeg in the worker): every pick captured.
    async with engine.connect() as conn:
        frames = await conn.scalar(
            select(stage_checkpoints.c.output).where(
                stage_checkpoints.c.job_id == job_id, stage_checkpoints.c.stage == "EXTRACT_FRAMES"
            )
        )
    assert frames is not None, "no EXTRACT_FRAMES checkpoint"
    assert frames["frames_requested"] == row["frames_requested"] > 0
    assert frames["frames_ok"] == frames["frames_requested"]
    assert frames["failed"] == []
    # Frame-aware notes (D34): SUMMARIZE showed the LLM real candidate frames (scdet + grabs).
    async with engine.connect() as conn:
        notes = await conn.scalar(
            select(stage_checkpoints.c.output).where(
                stage_checkpoints.c.job_id == job_id, stage_checkpoints.c.stage == "SUMMARIZE"
            )
        )
    assert notes is not None and notes["candidates"] >= frames["frames_requested"]
    # PUBLISH (CP8): one page, laid out as decided (video, link, summary, linked headings with
    # their frames, self-test toggles), every captured frame uploaded, the URL on the results.
    (page,) = await _notion_pages(job_id)
    assert page["blocks"][:3] == ["video", "callout", "paragraph"]
    assert page["blocks"].count("image") == frames["frames_ok"] == row["frames_published"]
    assert "toggle" in page["blocks"] and page["blocks"][-1] == "toggle"
    assert row["notion_page_id"] == page["id"]
    assert results.json()["result"]["notion_page_url"] == row["notion_page_url"]


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


async def _wait_for_page(job_id: UUID, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if await _notion_pages(job_id):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"no Notion page for job {job_id} after {timeout_s:g} s")


async def _checkpoint_output(engine: AsyncEngine, job_id: UUID, stage: Stage) -> dict[str, Any]:
    async with engine.connect() as conn:
        output: dict[str, Any] | None = await conn.scalar(
            select(stage_checkpoints.c.output).where(
                stage_checkpoints.c.job_id == job_id, stage_checkpoints.c.stage == stage.value
            )
        )
    assert output is not None, f"no {stage} checkpoint"
    return output


@pytest.mark.skipif(shutil.which("docker") is None, reason="needs the docker CLI")
@pytest.mark.parametrize("stage", STAGE_ORDER, ids=lambda stage: stage.value)
async def test_killing_the_worker_mid_stage_still_completes_the_job_once(
    gateway: httpx.AsyncClient, engine: AsyncEngine, job_rows: JobRows, stage: Stage
) -> None:
    job_id = await _submit(gateway, job_rows)

    # Wait until a worker runs the stage, then SIGKILL that worker's container (Spec §6).
    running = await wait_running(engine, stage, job_id, JOB_TIMEOUT_S)
    if stage is Stage.PUBLISH:
        # The riskiest moment for F-19: the page exists on Notion, the stage has not finished.
        await _wait_for_page(job_id, JOB_TIMEOUT_S)
    victim = running.worker  # container hostname = short container id
    killed_at = await kill(engine, victim)
    try:
        recovery = await follow(engine, job_id, stage, killed_at, JOB_TIMEOUT_S)
    finally:
        await docker("start", victim)

    assert recovery.status is JobStatus.COMPLETED
    assert recovery.redelivered_to not in (None, victim)
    events = await _events(engine, job_id)
    redelivered = [e for e in events if e["stage"] == stage.value and e["kind"] == "REDELIVERED"]
    assert redelivered, "the killed worker's message was not redelivered"
    assert redelivered[0]["detail"]["worker"] != victim
    assert redelivered[0]["detail"]["delivery_count"] >= 1
    # Each stage checkpointed exactly once, and the job finished exactly once.
    assert await _checkpoints(engine, job_id) == sorted(s.value for s in STAGE_ORDER)
    assert [t for t in _transitions(events) if t[1] == "COMPLETED"] == [("PUBLISHING", "COMPLETED")]
    # Exactly one Notion page, holding exactly the blocks PUBLISH built: none twice.
    (page,) = await _notion_pages(job_id)
    built = await _checkpoint_output(engine, job_id, Stage.PUBLISH)
    assert len(page["blocks"]) == built["blocks"]
    async with engine.connect() as conn:
        saved = await conn.scalar(select(jobs.c.notion_page_id).where(jobs.c.id == job_id))
    assert saved == page["id"]
