from typing import Any
from uuid import uuid4

import aio_pika
import pytest

from common.broker import (
    DLQ_QUEUE,
    EXCHANGE,
    WORK_QUEUE,
    BrokerPublisher,
    PublishFailedError,
    build_message,
    queue_specs,
)
from common.models import StageMessage
from common.state import Stage


def _message() -> StageMessage:
    return StageMessage(
        job_id=uuid4(), stage=Stage.SUMMARIZE, attempt=2, notion_token_enc="ciphertext"
    )


def test_every_queue_is_a_quorum_queue() -> None:
    specs = queue_specs([10, 30, 90])
    assert [spec.name for spec in specs] == [
        WORK_QUEUE,
        DLQ_QUEUE,
        "jobs.retry.10s",
        "jobs.retry.30s",
        "jobs.retry.90s",
    ]
    assert all(spec.arguments["x-queue-type"] == "quorum" for spec in specs)


def test_work_queue_dead_letters_to_dlq_so_the_delivery_limit_never_drops_a_message() -> None:
    work = queue_specs([10])[0]
    assert work.routing_key == "work"
    assert work.arguments["x-dead-letter-exchange"] == EXCHANGE
    assert work.arguments["x-dead-letter-routing-key"] == "dlq"


def test_retry_queues_expire_back_to_work_at_least_once() -> None:
    retry = {spec.name: spec for spec in queue_specs([2, 4, 8])}["jobs.retry.4s"]
    assert retry.routing_key == "retry.4s"
    assert retry.arguments == {
        "x-queue-type": "quorum",
        "x-message-ttl": 4000,
        "x-dead-letter-exchange": EXCHANGE,
        "x-dead-letter-routing-key": "work",
        "x-dead-letter-strategy": "at-least-once",
        "x-overflow": "reject-publish",
    }


def test_duplicate_delays_declare_one_queue() -> None:
    names = [spec.name for spec in queue_specs([30, 10, 30])]
    assert names.count("jobs.retry.30s") == 1


def test_messages_are_persistent_json_without_the_token_in_metadata() -> None:
    message = _message()
    built = build_message(message)
    assert built.delivery_mode == aio_pika.DeliveryMode.PERSISTENT
    assert built.content_type == "application/json"
    assert built.message_id == f"{message.job_id}:SUMMARIZE:2"
    assert StageMessage.from_body(built.body) == message
    assert "ciphertext" not in str(built.headers)


class _Exchange:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    # Mirrors aio-pika's Exchange.publish signature, `timeout` included.
    async def publish(
        self,
        message: aio_pika.Message,
        routing_key: str,
        timeout: float,  # noqa: ASYNC109
    ) -> None:
        self.calls.append({"message": message, "routing_key": routing_key, "timeout": timeout})
        if self.error is not None:
            raise self.error


async def test_publish_waits_for_confirm_with_the_configured_timeout() -> None:
    exchange = _Exchange()
    await BrokerPublisher(exchange, timeout_s=3.5).publish(_message())  # type: ignore[arg-type]
    assert exchange.calls[0]["routing_key"] == "work"
    assert exchange.calls[0]["timeout"] == 3.5


@pytest.mark.parametrize("error", [TimeoutError(), ConnectionError(), RuntimeError("nack")])
async def test_any_publish_failure_becomes_publish_failed(error: Exception) -> None:
    publisher = BrokerPublisher(_Exchange(error), timeout_s=1)  # type: ignore[arg-type]
    with pytest.raises(PublishFailedError):
        await publisher.publish(_message())
