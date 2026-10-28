"""Worker process entrypoint (ARCHITECTURE §7).

Until the runtime lands in M1 the worker only starts, logs, and waits for SIGTERM/SIGINT.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal

from common.config import get_settings
from common.logging import configure_logging, get_logger


async def run() -> None:
    log = get_logger()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):  # no signal handlers on Windows
            loop.add_signal_handler(sig, stop.set)
    log.info("worker_started", runtime="idle until M1")
    await stop.wait()
    log.info("worker_stopped")


def main() -> None:
    configure_logging(get_settings().log_level, service="worker")
    asyncio.run(run())


if __name__ == "__main__":
    main()
