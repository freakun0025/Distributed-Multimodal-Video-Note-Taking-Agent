"""REST API against the running stack (ARCHITECTURE §10; Spec §5)."""

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from common.broker import DLQ_QUEUE, WORK_QUEUE, PublishFailedError
from common.config import Settings
from common.crypto import TokenCipher, generate_key
from common.db import jobs
from common.models import StageMessage
from gateway.main import create_app
from tests.integration.conftest import VIDEO_ID, VIDEO_URL, JobRows

pytestmark = pytest.mark.integration

BODY = {"youtube_url": VIDEO_URL, "notion_token": "integration-test-token"}


@pytest.fixture
async def submit(gateway: httpx.AsyncClient, job_rows: JobRows) -> AsyncIterator[Any]:
    async def _submit(**kwargs: Any) -> httpx.Response:
        response = await gateway.post("/jobs", **kwargs)
        if response.status_code == 202:
            job_rows.track(response.json()["job_id"])
        return response

    yield _submit


async def test_post_returns_202_with_location_and_the_job_is_confirmed_enqueued(
    submit: Any, engine: AsyncEngine
) -> None:
    response = await submit(json=BODY)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "QUEUED"
    assert body["video_id"] == VIDEO_ID
    assert response.headers["location"] == f"/jobs/{body['job_id']}"
    async with engine.connect() as conn:
        row = (
            (await conn.execute(select(jobs).where(jobs.c.id == UUID(body["job_id"]))))
            .mappings()
            .one()
        )
    # enqueued_at is only set after the publisher confirm (outbox step 4).
    assert row["enqueued_at"] is not None
    assert row["youtube_url"] == VIDEO_URL
    assert "integration-test-token" not in str(dict(row))


async def test_get_job_reports_a_fresh_job_at_the_start_of_the_pipeline(
    submit: Any, gateway: httpx.AsyncClient
) -> None:
    job_id = (await submit(json=BODY)).json()["job_id"]
    response = await gateway.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert body["warnings"] == []
    assert body["error"] is None
    assert body["result"] is None
    assert body["waiting"] is None
    # Workers may already have claimed it: QUEUED at 0 %, or TRANSCRIBING within 5-20 %.
    assert (body["status"], body["progress"] == 0) in {("QUEUED", True), ("TRANSCRIBING", False)}
    assert 0 <= body["progress"] <= 20


async def test_results_are_409_until_the_job_is_terminal(
    submit: Any, gateway: httpx.AsyncClient
) -> None:
    job_id = (await submit(json=BODY)).json()["job_id"]
    response = await gateway.get(f"/jobs/{job_id}/results")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "JOB_NOT_TERMINAL"


@pytest.mark.parametrize("suffix", ["", "/results"])
async def test_unknown_job_is_404(gateway: httpx.AsyncClient, suffix: str) -> None:
    response = await gateway.get(f"/jobs/{uuid4()}{suffix}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "JOB_NOT_FOUND"


async def test_same_idempotency_key_returns_the_same_job(submit: Any, engine: AsyncEngine) -> None:
    key = f"itest-{uuid4()}"
    first = await submit(json=BODY, headers={"Idempotency-Key": key})
    second = await submit(json=BODY, headers={"Idempotency-Key": key})
    assert first.status_code == second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]
    async with engine.connect() as conn:
        count = await conn.scalar(
            select(func.count()).select_from(jobs).where(jobs.c.idempotency_key == key)
        )
    assert count == 1


async def test_broker_topology_has_quorum_queues_and_a_dead_letter_route(
    rabbitmq_api: httpx.AsyncClient,
) -> None:
    work = (await rabbitmq_api.get(f"/queues/%2F/{WORK_QUEUE}")).json()
    assert work["type"] == "quorum"
    assert work["arguments"]["x-dead-letter-routing-key"] == "dlq"
    assert (await rabbitmq_api.get(f"/queues/%2F/{DLQ_QUEUE}")).status_code == 200
    for delay in Settings().retry_delays:
        retry = (await rabbitmq_api.get(f"/queues/%2F/jobs.retry.{delay}s")).json()
        assert retry["arguments"]["x-message-ttl"] == delay * 1000
        assert retry["arguments"]["x-dead-letter-routing-key"] == "work"
        assert retry["arguments"]["x-dead-letter-strategy"] == "at-least-once"


class _RefusingPublisher:
    def __init__(self) -> None:
        self.messages: list[StageMessage] = []

    async def publish(self, message: StageMessage, routing_key: str = "work") -> None:
        self.messages.append(message)
        raise PublishFailedError("broker down")


async def test_unconfirmed_publish_fails_the_job_and_returns_503(
    engine: AsyncEngine, job_rows: JobRows
) -> None:
    app = create_app(Settings(_env_file=None))  # type: ignore[call-arg]
    publisher = _RefusingPublisher()
    app.state.engine = engine
    app.state.publisher = publisher
    app.state.cipher = TokenCipher(generate_key())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/jobs", json=BODY)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "BROKER_UNAVAILABLE"
    [message] = publisher.messages
    job_rows.track(message.job_id)
    assert message.notion_token_enc != BODY["notion_token"]  # encrypted, never plaintext
    async with engine.connect() as conn:
        row = (await conn.execute(select(jobs).where(jobs.c.id == message.job_id))).mappings().one()
    assert row["status"] == "FAILED"
    assert row["error"]["code"] == "ENQUEUE_FAILED"
    assert row["enqueued_at"] is None
