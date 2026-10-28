"""FastAPI app factory and lifespan: DB, broker, outbox sweeper (ARCHITECTURE §10).

The gateway never runs pipeline work (invariant 1).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from common.config import Settings, get_settings
from common.db import create_engine
from common.logging import get_logger
from gateway.routes import health

log = get_logger()


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(app_settings.database_url)
        redis = Redis.from_url(app_settings.redis_url)
        app.state.health_checks = health.default_checks(
            engine, redis, app_settings.rabbitmq_url, app_settings.healthcheck_timeout_s
        )
        log.info("gateway_started")
        try:
            yield
        finally:
            await redis.aclose()
            await engine.dispose()
            log.info("gateway_stopped")

    app = FastAPI(title="Video Notes Agent — gateway", version="0.1.0", lifespan=lifespan)
    app.state.settings = app_settings
    app.include_router(health.router)
    return app


app = create_app()
