"""worker/llm/health.py against the stack's real Redis (published on 127.0.0.1:6380), D36."""

import asyncio
import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from redis.asyncio import Redis

from common.errors import ErrorCode, RetryableError
from worker.llm.health import RedisModelHealth

pytestmark = pytest.mark.integration

REDIS_URL = os.environ.get("ITEST_REDIS_URL", "redis://127.0.0.1:6380/0")
MODEL, OTHER = "model-a", "model-b"


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    client = Redis.from_url(REDIS_URL)
    yield client
    await client.aclose()


@pytest.fixture
def prefix() -> str:
    return f"llm-test:{uuid4().hex}:"


def _health(redis: Redis, prefix: str, **overrides: float) -> RedisModelHealth:
    values = {"strikes": 2, "strike_window_s": 60, "cooldown_s": 30, "quota_reset_s": 5000.0}
    values.update(overrides)
    reset = values.pop("quota_reset_s")
    return RedisModelHealth(
        redis,
        strikes=int(values["strikes"]),
        strike_window_s=values["strike_window_s"],
        cooldown_s=values["cooldown_s"],
        quota_reset_s=lambda: reset,
        key_prefix=prefix,
    )


async def test_the_second_strike_benches_the_model_for_the_cooldown(
    redis: Redis, prefix: str
) -> None:
    health = _health(redis, prefix)
    assert await health.overloaded(MODEL) is False
    assert await health.benched([MODEL, OTHER]) == {}
    assert await health.overloaded(MODEL) is True
    benched = await health.benched([MODEL, OTHER])
    assert list(benched) == [MODEL]
    assert 29 < benched[MODEL] <= 30
    assert await redis.exists(health.strikes_key(MODEL)) == 0  # counting starts afresh


async def test_strikes_expire_with_their_window(redis: Redis, prefix: str) -> None:
    health = _health(redis, prefix, strike_window_s=0.2)
    assert await health.overloaded(MODEL) is False
    await asyncio.sleep(0.4)
    assert await health.overloaded(MODEL) is False  # the first strike has expired


async def test_an_answer_clears_the_strikes(redis: Redis, prefix: str) -> None:
    health = _health(redis, prefix)
    await health.overloaded(MODEL)
    await health.answered(MODEL)
    assert await health.overloaded(MODEL) is False


async def test_a_spent_daily_quota_benches_until_the_reset(redis: Redis, prefix: str) -> None:
    health = _health(redis, prefix)
    assert await health.quota_spent(MODEL) == 5000
    assert 4999 < (await health.benched([MODEL]))[MODEL] <= 5000


async def test_strikes_from_workers_failing_at_once_are_counted_atomically(
    redis: Redis, prefix: str
) -> None:
    clients = [Redis.from_url(REDIS_URL) for _ in range(4)]  # four workers
    try:
        healths = [_health(client, prefix) for client in clients]
        benched = await asyncio.gather(*(health.overloaded(MODEL) for health in healths))
    finally:
        await asyncio.gather(*(client.aclose() for client in clients))
    # No strike lost or double-counted: every second one benches (2 strikes per bench).
    assert sorted(benched) == [False, False, True, True]
    assert MODEL in await _health(redis, prefix).benched([MODEL])


async def test_every_worker_sees_a_bench(redis: Redis, prefix: str) -> None:
    other_worker = Redis.from_url(REDIS_URL)
    try:
        await _health(redis, prefix).quota_spent(MODEL)
        assert MODEL in await _health(other_worker, prefix).benched([MODEL, OTHER])
    finally:
        await other_worker.aclose()


async def test_unreachable_redis_fails_closed() -> None:
    client = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.5)
    try:
        with pytest.raises(RetryableError) as caught:
            await _health(client, "llm-test:").benched([MODEL])
        assert caught.value.code is ErrorCode.LIMITER_UNAVAILABLE
    finally:
        await client.aclose()
