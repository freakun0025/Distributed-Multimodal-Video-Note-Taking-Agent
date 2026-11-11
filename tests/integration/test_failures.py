"""Spec §6 failure modes and Constraint A end to end (CP9): the fakes service's switches make an
external API answer 429, a video private or too long, or frames fail; the real gateway, broker,
Postgres and workers must keep the job waiting (not failed), dead-letter it after the budget,
reject it at once, or publish what worked.

Needs `make up-fakes`. Reads RETRY_DELAYS / RETRY_MAX_ATTEMPTS from the repo's .env like the
workers do, so the expected delays and the DLQ test's time limit follow the stack's settings.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from common.broker import DLQ_QUEUE, retry_delay
from common.config import Settings
from common.state import TERMINAL_STATUSES
from fakes.app import CUT_AT_S, PRIVATE_VIDEO
from tests.integration.conftest import FAKES_URL, JobRows
from tests.integration.test_pipeline import (
    JOB_TIMEOUT_S,
    _events,
    _notion_pages,
    _submit,
    _transitions,
)

pytestmark = pytest.mark.integration

SETTINGS = Settings()
DELAYS = SETTINGS.retry_delays
STAGE_OF = {"youtube": "TRANSCRIBE", "llm": "SUMMARIZE", "notion": "PUBLISH"}


@pytest.fixture
async def fakes() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=FAKES_URL, timeout=10) as client:
        await client.post("/_control/reset")
        yield client
        await client.post("/_control/reset")


async def _follow(
    gateway: httpx.AsyncClient, job_id: UUID, timeout_s: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Poll like the popup until the job ends: the final status and every response seen."""
    seen: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body: dict[str, Any] = (await gateway.get(f"/jobs/{job_id}")).json()
        seen.append(body)
        if body["status"] in TERMINAL_STATUSES:
            return body, seen
        await asyncio.sleep(0.5)
    raise AssertionError(f"job {job_id} not terminal after {timeout_s:g}s: {seen[-1]}")


def _shown_waiting(seen: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The responses in which the popup shows "Waiting — ...": never an error meanwhile."""
    waiting = [body for body in seen if body["waiting"]]
    assert all(body["error"] is None for body in waiting)
    assert all(body["status"] not in TERMINAL_STATUSES for body in waiting)
    return waiting


async def _retries(engine: AsyncEngine, job_id: UUID) -> list[tuple[str, str, int]]:
    return [
        (event["stage"], event["detail"]["code"], event["detail"]["delay_s"])
        for event in await _events(engine, job_id)
        if event["kind"] == "RETRY_SCHEDULED"
    ]


async def _statuses(fakes: httpx.AsyncClient, api: str) -> list[int]:
    calls = (await fakes.get("/_control/calls", params={"api": api})).json()
    return [call["status"] for call in calls]


# --- External rate limits (HTTP 429) on any of the three APIs ------------------------------------


@pytest.mark.parametrize("api", ["youtube", "llm", "notion"])
async def test_a_429_from_any_api_makes_the_job_wait_then_complete(
    api: str,
    gateway: httpx.AsyncClient,
    engine: AsyncEngine,
    job_rows: JobRows,
    fakes: httpx.AsyncClient,
) -> None:
    await fakes.put("/_control/faults", json={"api": api, "status": 429})
    job_id = await _submit(gateway, job_rows)
    final, seen = await _follow(gateway, job_id, JOB_TIMEOUT_S + DELAYS[0])

    assert final["status"] == "COMPLETED", final
    waiting = _shown_waiting(seen)
    assert waiting, "the job was never shown waiting"
    assert {body["waiting"]["reason"] for body in waiting} == {f"RATE_LIMITED:{api}"}
    # Re-queued once through the first delay queue, then the same stage ran again.
    assert await _retries(engine, job_id) == [(STAGE_OF[api], "RATE_LIMITED", DELAYS[0])]
    statuses = await _statuses(fakes, api)
    assert statuses[0] == 429 and all(status < 400 for status in statuses[1:])
    assert len(await _notion_pages(job_id)) == 1


async def test_repeated_429s_back_off_through_the_growing_delays(
    gateway: httpx.AsyncClient, engine: AsyncEngine, job_rows: JobRows, fakes: httpx.AsyncClient
) -> None:
    await fakes.put("/_control/faults", json={"api": "notion", "count": 2})
    job_id = await _submit(gateway, job_rows)
    expected = [retry_delay(DELAYS, attempt, None) for attempt in (1, 2)]  # 10 s, then 30 s
    final, seen = await _follow(gateway, job_id, JOB_TIMEOUT_S + sum(expected))

    assert final["status"] == "COMPLETED", final
    assert _shown_waiting(seen)
    assert await _retries(engine, job_id) == [("PUBLISH", "RATE_LIMITED", d) for d in expected]
    assert len(await _notion_pages(job_id)) == 1  # the retried PUBLISH found its own page


async def test_a_retry_after_longer_than_the_backoff_is_honoured(
    gateway: httpx.AsyncClient, engine: AsyncEngine, job_rows: JobRows, fakes: httpx.AsyncClient
) -> None:
    asked = DELAYS[0] + 5  # Google's RetryInfo.retryDelay in the 429 body
    await fakes.put("/_control/faults", json={"api": "llm", "retry_after": asked})
    job_id = await _submit(gateway, job_rows)
    expected = retry_delay(DELAYS, 1, asked)  # rounded up to a delay queue: 30 s
    final, _ = await _follow(gateway, job_id, JOB_TIMEOUT_S + expected)

    assert final["status"] == "COMPLETED", final
    assert expected >= asked
    assert await _retries(engine, job_id) == [("SUMMARIZE", "RATE_LIMITED", expected)]


# --- Dead-letter queue: a private video fails every attempt ------------------------------------


async def _dlq_depth(rabbitmq_api: httpx.AsyncClient) -> int:
    depth: int = (await rabbitmq_api.get(f"/queues/%2F/{DLQ_QUEUE}")).json().get("messages", 0)
    return depth


async def test_an_unavailable_video_is_retried_then_dead_lettered(
    gateway: httpx.AsyncClient,
    engine: AsyncEngine,
    job_rows: JobRows,
    fakes: httpx.AsyncClient,
    rabbitmq_api: httpx.AsyncClient,
) -> None:
    attempts = SETTINGS.retry_max_attempts
    expected = [retry_delay(DELAYS, attempt, None) for attempt in range(1, attempts)]
    before = await _dlq_depth(rabbitmq_api)
    await fakes.put("/_control/youtube", json={"unavailable": True})
    started = time.monotonic()
    job_id = await _submit(gateway, job_rows)
    final, seen = await _follow(gateway, job_id, JOB_TIMEOUT_S + sum(expected))
    took_s = time.monotonic() - started

    assert final["status"] == "DEAD_LETTERED", final
    assert final["error"]["code"] == "VIDEO_UNAVAILABLE"
    assert PRIVATE_VIDEO in final["error"]["message"]
    waiting = _shown_waiting(seen)  # retries remained: waiting, never an error
    assert {body["waiting"]["reason"] for body in waiting} == {"VIDEO_UNAVAILABLE:youtube"}
    assert await _retries(engine, job_id) == [
        ("TRANSCRIBE", "VIDEO_UNAVAILABLE", delay) for delay in expected
    ]
    entered = [e["attempt"] for e in await _events(engine, job_id) if e["kind"] == "TRANSITION"]
    assert entered[:-1] == list(range(1, attempts + 1))  # each attempt entered TRANSCRIBING
    assert _transitions(await _events(engine, job_id))[-1] == ("TRANSCRIBING", "DEAD_LETTERED")
    results = await gateway.get(f"/jobs/{job_id}/results")
    assert results.status_code == 200
    assert results.json()["status"] == "DEAD_LETTERED" and results.json()["result"] is None
    assert await _notion_pages(job_id) == []
    # The message itself waits in jobs.dlq for inspection (management stats lag a few seconds).
    depth = before
    for _ in range(30):
        if (depth := await _dlq_depth(rabbitmq_api)) > before:
            break
        await asyncio.sleep(1)
    assert depth > before
    # Bounded and demonstrable: the delays plus a few quick attempts.
    assert took_s < sum(expected) + 60, f"took {took_s:.0f} s"


# --- Graceful degradation: frames that fail --------------------------------------------------


async def _results(gateway: httpx.AsyncClient, job_id: UUID) -> dict[str, Any]:
    response = await gateway.get(f"/jobs/{job_id}/results")
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    return body


async def test_when_some_frames_fail_the_page_keeps_the_others_with_warnings(
    gateway: httpx.AsyncClient, job_rows: JobRows, fakes: httpx.AsyncClient
) -> None:
    await fakes.put("/_control/youtube", json={"frames": "partial"})  # video cut at 5:00
    job_id = await _submit(gateway, job_rows)
    final, seen = await _follow(gateway, job_id, JOB_TIMEOUT_S)

    assert final["status"] == "COMPLETED_WITH_WARNINGS", final
    results = await _results(gateway, job_id)
    result, warnings = results["result"], results["warnings"]
    assert 1 <= result["frames_published"] < result["frames_requested"]
    assert len(warnings) == result["frames_requested"] - result["frames_published"]
    assert {w["code"] for w in warnings} == {"FRAME_EXTRACTION_FAILED"}
    # Only the picks the cut-short video no longer has (each burst grab stays inside its second).
    assert all(w["timestamp_sec"] >= CUT_AT_S - 1 for w in warnings), warnings
    assert any(body["warnings"] for body in seen)  # the popup lists them too
    (page,) = await _notion_pages(job_id)
    assert page["blocks"].count("image") == result["frames_published"]
    assert "heading_2" in page["blocks"] and "bulleted_list_item" in page["blocks"]


async def test_when_every_frame_fails_the_page_is_text_only(
    gateway: httpx.AsyncClient, job_rows: JobRows, fakes: httpx.AsyncClient
) -> None:
    await fakes.put("/_control/youtube", json={"frames": "none"})  # a black video
    job_id = await _submit(gateway, job_rows)
    final, _ = await _follow(gateway, job_id, JOB_TIMEOUT_S)

    assert final["status"] == "COMPLETED_WITH_WARNINGS", final
    results = await _results(gateway, job_id)
    result, warnings = results["result"], results["warnings"]
    assert result["frames_requested"] > 0 and result["frames_published"] == 0
    assert len(warnings) == result["frames_requested"]
    assert {w["message"] for w in warnings} == {"blank frame (black)"}
    (page,) = await _notion_pages(job_id)
    assert "image" not in page["blocks"]  # never substituted (C-B)
    assert page["blocks"][:3] == ["video", "callout", "paragraph"]
    assert "heading_2" in page["blocks"] and page["blocks"][-1] == "toggle"


# --- Constraint A: longer than 20 minutes is rejected, nothing processed (Spec §7) -----------


async def test_a_video_over_the_limit_fails_without_any_processing(
    gateway: httpx.AsyncClient,
    engine: AsyncEngine,
    job_rows: JobRows,
    fakes: httpx.AsyncClient,
) -> None:
    await fakes.put("/_control/youtube", json={"duration": SETTINGS.max_video_seconds + 300})
    job_id = await _submit(gateway, job_rows)
    final, _ = await _follow(gateway, job_id, JOB_TIMEOUT_S)

    assert final["status"] == "FAILED", final
    assert final["error"]["code"] == "VIDEO_TOO_LONG"
    assert _transitions(await _events(engine, job_id)) == [
        ("QUEUED", "TRANSCRIBING"),
        ("TRANSCRIBING", "FAILED"),
    ]
    assert await _retries(engine, job_id) == []  # permanent: never retried
    assert await _statuses(fakes, "llm") == []  # no LLM call, no frames, no page
    assert await _notion_pages(job_id) == []
