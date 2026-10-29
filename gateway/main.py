"""FastAPI app factory and lifespan: DB, broker, outbox sweeper (ARCHITECTURE §10).

The gateway never runs pipeline work (invariant 1).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from common.broker import BrokerPublisher
from common.config import Settings, get_settings
from common.crypto import TokenCipher
from common.db import create_engine
from common.logging import get_logger
from gateway.errors import install_error_handlers
from gateway.routes import health, jobs
from gateway.sweeper import run_sweeper

log = get_logger()


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app_settings.token_encryption_key is None:
            raise RuntimeError("TOKEN_ENCRYPTION_KEY is not set (make fernet-key)")
        app.state.cipher = TokenCipher(app_settings.token_encryption_key)
        engine = create_engine(app_settings.database_url)
        redis = Redis.from_url(app_settings.redis_url)
        publisher = await BrokerPublisher.connect(
            app_settings.rabbitmq_url,
            app_settings.retry_delays,
            app_settings.broker_publish_timeout_s,
        )
        app.state.engine = engine
        app.state.publisher = publisher
        app.state.health_checks = health.default_checks(
            engine, redis, app_settings.rabbitmq_url, app_settings.healthcheck_timeout_s
        )
        sweeper = asyncio.create_task(
            run_sweeper(engine, app_settings.outbox_sweep_interval_s, app_settings.outbox_timeout_s)
        )
        log.info("gateway_started")
        try:
            yield
        finally:
            sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper
            await publisher.close()
            await redis.aclose()
            await engine.dispose()
            log.info("gateway_stopped")

    app = FastAPI(title="Video Notes Agent — gateway", version="0.1.0", lifespan=lifespan)
    app.state.settings = app_settings
    install_error_handlers(app)
    app.include_router(jobs.router)
    app.include_router(health.router)
    return app


app = create_app()
