"""Request validation and response shaping; DB-backed paths are in tests/integration."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import httpx
import pytest

from common.config import Settings
from common.state import JobStatus
from gateway.main import create_app
from gateway.routes.jobs import job_status_response

VALID_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


async def _request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    app = create_app(Settings(_env_file=None))  # type: ignore[call-arg]
    app.state.engine = None  # validation fails before any DB access
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


@pytest.mark.parametrize(
    "url", ["https://vimeo.com/123", "not a url", "https://www.youtube.com/watch?v=short"]
)
async def test_post_rejects_non_youtube_urls_with_invalid_url(url: str) -> None:
    response = await _request("POST", "/jobs", json={"youtube_url": url, "notion_token": "t"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_URL"


@pytest.mark.parametrize("body", [{}, {"notion_token": None}, {"notion_token": "   "}])
async def test_post_without_a_notion_token_is_400_missing_notion_token(
    body: dict[str, Any],
) -> None:
    response = await _request("POST", "/jobs", json={"youtube_url": VALID_URL, **body})
    assert response.status_code == 400
    assert response.json() == {
        "error": {"code": "MISSING_NOTION_TOKEN", "message": "notion_token is required"}
    }


async def test_schema_errors_use_the_error_body_and_never_echo_the_token() -> None:
    response = await _request(
        "POST", "/jobs", json={"notion_token": "very-secret-token", "youtube_url": 42}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "youtube_url" in response.json()["error"]["message"]
    assert "very-secret-token" not in response.text


async def test_malformed_json_is_a_validation_error() -> None:
    response = await _request(
        "POST", "/jobs", content=b"{", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize("path", ["/jobs/not-a-uuid", "/jobs/not-a-uuid/results"])
async def test_malformed_job_id_is_404_job_not_found(path: str) -> None:
    response = await _request("GET", path)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "JOB_NOT_FOUND"


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": uuid4(),
        "status": "QUEUED",
        "progress": 0,
        "video_title": None,
        "duration_sec": None,
        "warnings": [],
        "error": None,
        "waiting_until": None,
        "waiting_reason": None,
        "notion_page_url": None,
        "frames_requested": None,
        "frames_published": None,
    }
    return row | overrides


def test_completed_job_reports_its_notion_page_and_frame_counts() -> None:
    response = job_status_response(
        _row(
            status="COMPLETED_WITH_WARNINGS",
            progress=100,
            notion_page_url="https://www.notion.so/page",
            frames_requested=6,
            frames_published=5,
            warnings=[{"code": "FRAME_EXTRACTION_FAILED", "timestamp_sec": 461}],
        )
    )
    assert response.result is not None
    assert response.result.frames_published == 5
    dumped = response.model_dump(mode="json")
    assert dumped["warnings"] == [{"code": "FRAME_EXTRACTION_FAILED", "timestamp_sec": 461}]


def test_failed_job_reports_error_and_no_result() -> None:
    response = job_status_response(
        _row(status="FAILED", error={"code": "VIDEO_TOO_LONG", "message": "too long"})
    )
    assert response.status is JobStatus.FAILED
    assert response.result is None
    assert response.error is not None
    assert response.error.code == "VIDEO_TOO_LONG"


def test_waiting_is_shown_only_while_the_job_is_running() -> None:
    until = datetime.now(UTC) + timedelta(seconds=30)
    running = job_status_response(
        _row(status="SUMMARIZING", waiting_until=until, waiting_reason="RATE_LIMITED:llm")
    )
    assert running.waiting is not None
    assert running.waiting.reason == "RATE_LIMITED:llm"
    done = job_status_response(
        _row(
            status="DEAD_LETTERED",
            waiting_until=until,
            waiting_reason="RATE_LIMITED:llm",
            error={"code": "RATE_LIMITED", "message": "x"},
        )
    )
    assert done.waiting is None
