"""Exchange/queue declarations and confirmed, persistent publishing (ARCHITECTURE §6.1).

Declaration and publishing only; consuming lives in `worker/runtime.py`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractRobustConnection

from common.logging import get_logger
from common.models import StageMessage

EXCHANGE = "vna"
WORK_QUEUE = "jobs.work"
WORK_ROUTING_KEY = "work"
DLQ_QUEUE = "jobs.dlq"
DLQ_ROUTING_KEY = "dlq"

log = get_logger()


class PublishFailedError(Exception):
    """The broker did not confirm a publish (down, nack, unroutable or timed out)."""


@dataclass(frozen=True)
class QueueSpec:
    name: str
    routing_key: str
    arguments: dict[str, Any]


def retry_queue_name(delay_s: int) -> str:
    return f"jobs.retry.{delay_s}s"


def retry_routing_key(delay_s: int) -> str:
    return f"retry.{delay_s}s"


def queue_specs(retry_delays: Iterable[int]) -> list[QueueSpec]:
    """Every queue in the topology. Quorum queues throughout (ARCHITECTURE §6.1).

    `jobs.work` dead-letters to `jobs.dlq`: RabbitMQ 4 drops a message past the delivery limit
    unless the queue has a DLX. Retry queues hold a message for their TTL, then dead-letter it
    back to `jobs.work` at-least-once (which RabbitMQ only allows with `reject-publish`).
    """
    quorum: dict[str, Any] = {"x-queue-type": "quorum"}
    specs = [
        QueueSpec(
            WORK_QUEUE,
            WORK_ROUTING_KEY,
            {
                **quorum,
                "x-dead-letter-exchange": EXCHANGE,
                "x-dead-letter-routing-key": DLQ_ROUTING_KEY,
            },
        ),
        QueueSpec(DLQ_QUEUE, DLQ_ROUTING_KEY, dict(quorum)),
    ]
    for delay_s in sorted(set(retry_delays)):
        specs.append(
            QueueSpec(
                retry_queue_name(delay_s),
                retry_routing_key(delay_s),
                {
                    **quorum,
                    "x-message-ttl": delay_s * 1000,
                    "x-dead-letter-exchange": EXCHANGE,
                    "x-dead-letter-routing-key": WORK_ROUTING_KEY,
                    "x-dead-letter-strategy": "at-least-once",
                    "x-overflow": "reject-publish",
                },
            )
        )
    return specs


async def declare_topology(
    channel: AbstractChannel, retry_delays: Sequence[int]
) -> AbstractExchange:
    """Idempotently declare the exchange, every queue and its binding; return the exchange."""
    exchange = await channel.declare_exchange(EXCHANGE, aio_pika.ExchangeType.DIRECT, durable=True)
    for spec in queue_specs(retry_delays):
        queue = await channel.declare_queue(spec.name, durable=True, arguments=spec.arguments)
        await queue.bind(exchange, routing_key=spec.routing_key)
    return exchange


def build_message(message: StageMessage) -> aio_pika.Message:
    return aio_pika.Message(
        body=message.to_body(),
        content_type="application/json",
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        message_id=f"{message.job_id}:{message.stage}:{message.attempt}",
    )


class BrokerPublisher:
    """Persistent publishes that return only once RabbitMQ has confirmed them."""

    def __init__(
        self,
        exchange: AbstractExchange,
        timeout_s: float,
        connection: AbstractRobustConnection | None = None,
    ) -> None:
        self._exchange = exchange
        self._timeout_s = timeout_s
        self._connection = connection

    @classmethod
    async def connect(
        cls, url: str, retry_delays: Sequence[int], timeout_s: float
    ) -> BrokerPublisher:
        connection = await aio_pika.connect_robust(url)
        # on_return_raises: an unroutable message is an error, not a silent drop.
        channel = await connection.channel(publisher_confirms=True, on_return_raises=True)
        exchange = await declare_topology(channel, retry_delays)
        return cls(exchange, timeout_s, connection)

    async def publish(self, message: StageMessage, routing_key: str = WORK_ROUTING_KEY) -> None:
        try:
            await self._exchange.publish(
                build_message(message), routing_key=routing_key, timeout=self._timeout_s
            )
        except Exception as exc:
            log.warning(
                "publish_failed",
                job_id=str(message.job_id),
                stage=message.stage,
                attempt=message.attempt,
                routing_key=routing_key,
                error=type(exc).__name__,
            )
            raise PublishFailedError(type(exc).__name__) from exc

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
