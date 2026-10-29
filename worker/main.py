"""Worker process entrypoint: consume jobs.work one message at a time (ARCHITECTURE §6.1, §7).

SIGTERM/SIGINT stops consuming and lets the in-flight message finish; if the container is
killed first, the unacked message is redelivered to another worker.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import socket

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from common.broker import WORK_QUEUE, BrokerPublisher, declare_topology
from common.config import get_settings
from common.db import create_engine
from common.logging import configure_logging, get_logger
from worker.runtime import Runtime
from worker.stages import STAGES
from worker.store import PostgresJobStore

# One unacked message per worker: a stage holds the worker until it acks (ARCHITECTURE §6.1).
PREFETCH_COUNT = 1


async def run() -> None:
    log = get_logger()
    settings = get_settings()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):  # no signal handlers on Windows
            loop.add_signal_handler(sig, stop.set)

    engine = create_engine(settings.database_url)
    connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    try:
        channel = await connection.channel(publisher_confirms=True, on_return_raises=True)
        await channel.set_qos(prefetch_count=PREFETCH_COUNT)
        exchange = await declare_topology(channel, settings.retry_delays)
        queue = await channel.get_queue(WORK_QUEUE)
        runtime = Runtime(
            PostgresJobStore(engine),
            BrokerPublisher(exchange, settings.broker_publish_timeout_s),
            settings,
            STAGES,
            worker=socket.gethostname(),
        )

        idle = asyncio.Event()
        idle.set()

        async def on_message(message: AbstractIncomingMessage) -> None:
            idle.clear()
            try:
                await runtime.handle(message)
            finally:
                idle.set()

        consumer_tag = await queue.consume(on_message)
        log.info("worker_started", queue=WORK_QUEUE, prefetch=PREFETCH_COUNT)
        await stop.wait()
        log.info("worker_stopping")
        await queue.cancel(consumer_tag)
        await idle.wait()
    finally:
        await connection.close()
        await engine.dispose()
    log.info("worker_stopped")


def main() -> None:
    configure_logging(get_settings().log_level, service="worker")
    asyncio.run(run())


if __name__ == "__main__":
    main()
