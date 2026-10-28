"""GET /healthz: postgres, rabbitmq, redis reachability (ARCHITECTURE §10; Spec §8)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping

import aio_pika
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from common.logging import get_logger
from common.models import ComponentHealth, HealthResponse

HealthCheck = Callable[[], Awaitable[None]]

router = APIRouter()
log = get_logger()


def default_checks(
    engine: AsyncEngine, redis: Redis, rabbitmq_url: str, timeout_s: float
) -> dict[str, HealthCheck]:
    async def postgres() -> None:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def rabbitmq() -> None:
        connection = await aio_pika.connect(rabbitmq_url, timeout=timeout_s)
        await connection.close()

    async def redis_ping() -> None:
        await redis.ping()

    return {"postgres": postgres, "rabbitmq": rabbitmq, "redis": redis_ping}


async def _status(name: str, check: HealthCheck, timeout_s: float) -> ComponentHealth:
    try:
        await asyncio.wait_for(check(), timeout=timeout_s)
    except Exception as exc:
        log.warning("healthcheck_failed", component=name, error=type(exc).__name__)
        return "down"
    return "ok"


async def run_checks(checks: Mapping[str, HealthCheck], timeout_s: float) -> HealthResponse:
    names = ("postgres", "rabbitmq", "redis")
    results = await asyncio.gather(*(_status(name, checks[name], timeout_s) for name in names))
    return HealthResponse(postgres=results[0], rabbitmq=results[1], redis=results[2])


@router.get(
    "/healthz",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "A dependency is unreachable"}},
)
async def healthz(request: Request) -> JSONResponse:
    state = request.app.state
    health = await run_checks(state.health_checks, state.settings.healthcheck_timeout_s)
    healthy = all(value == "ok" for value in (health.postgres, health.rabbitmq, health.redis))
    return JSONResponse(health.model_dump(), status_code=200 if healthy else 503)
