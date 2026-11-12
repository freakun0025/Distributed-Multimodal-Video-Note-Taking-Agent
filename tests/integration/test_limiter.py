"""common/ratelimit.lua against the stack's real Redis (published on 127.0.0.1:6380).

Assertions use the waits Redis books (computed on Redis time), not wall-clock grant times, so
event-loop timer jitter can't make them flaky.
"""

import asyncio
import itertools
import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from redis.asyncio import Redis

from common.errors import ErrorCode, ExternalApi, RetryableError
from common.limiter import Bucket, RateLimiter

pytestmark = pytest.mark.integration

REDIS_URL = os.environ.get("ITEST_REDIS_URL", "redis://127.0.0.1:6380/0")
TOLERANCE_S = 0.03  # time between requests issued "together", plus rounding to whole ms


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    client = Redis.from_url(REDIS_URL)
    yield client
    await client.aclose()


@pytest.fixture
def prefix() -> str:
    return f"rl-test:{uuid4().hex}:"


def _limiter(redis: Redis, prefix: str, bucket: Bucket, max_wait_s: float = 5) -> RateLimiter:
    return RateLimiter(redis, {ExternalApi.LLM: bucket}, max_wait_s, key_prefix=prefix)


async def _redis_now(redis: Redis) -> float:
    seconds, micros = await redis.time()
    return float(seconds) + micros / 1_000_000


async def _idle_on_redis_clock(redis: Redis, seconds: float) -> None:
    """Let `seconds` pass on Redis's clock, the one the bucket refills by. Redis runs in Docker's
    VM: one itest run saw its clock advance ~1 ms while the test slept 0.5 s on the host."""
    until = await _redis_now(redis) + seconds
    for _ in range(100):
        left = until - await _redis_now(redis)
        if left <= 0:
            return
        await asyncio.sleep(left)
    raise AssertionError(f"Redis's clock did not advance {seconds:g} s")


async def test_the_burst_passes_at_once_then_calls_are_spaced_at_the_rate(
    redis: Redis, prefix: str
) -> None:
    limiter = _limiter(redis, prefix, Bucket(rate_per_s=10, capacity=2))
    waits = [await limiter.acquire(ExternalApi.LLM) for _ in range(5)]
    assert waits[:2] == [0, 0]
    for wait in waits[2:]:
        assert 0.1 - TOLERANCE_S <= wait <= 0.1


async def test_workers_sharing_redis_are_served_in_turn_at_the_rate(
    redis: Redis, prefix: str
) -> None:
    # Three limiters on three connections = three workers; eight calls arrive at once.
    clients = [Redis.from_url(REDIS_URL) for _ in range(3)]
    try:
        limiters = [_limiter(c, prefix, Bucket(rate_per_s=5, capacity=1)) for c in clients]
        waits = await asyncio.gather(*(limiters[i % 3].acquire(ExternalApi.LLM) for i in range(8)))
    finally:
        await asyncio.gather(*(c.aclose() for c in clients))
    booked = sorted(waits)
    assert booked[0] == 0
    # One slot every 1/rate seconds, no two calls in the same slot.
    for earlier, later in itertools.pairwise(booked):
        assert 0.2 - TOLERANCE_S <= later - earlier <= 0.2 + TOLERANCE_S


async def test_a_wait_longer_than_max_wait_books_nothing(redis: Redis, prefix: str) -> None:
    limiter = _limiter(redis, prefix, Bucket(rate_per_s=1, capacity=1), max_wait_s=0.5)
    assert await limiter.acquire(ExternalApi.LLM) == 0
    for _ in range(2):  # the second refusal still sees a 1 s wait: the first booked nothing
        with pytest.raises(RetryableError) as caught:
            await limiter.acquire(ExternalApi.LLM)
        assert caught.value.code is ErrorCode.RATE_LIMITED_LOCAL
        assert caught.value.retry_after_s is not None
        assert 1 - TOLERANCE_S <= caught.value.retry_after_s <= 1


async def test_an_idle_bucket_refills_to_capacity_and_no_further(redis: Redis, prefix: str) -> None:
    limiter = _limiter(redis, prefix, Bucket(rate_per_s=10, capacity=2))
    for _ in range(2):
        await limiter.acquire(ExternalApi.LLM)
    await _idle_on_redis_clock(redis, 0.5)  # enough for 5 tokens; the bucket holds only 2
    waits = [await limiter.acquire(ExternalApi.LLM) for _ in range(3)]
    assert waits[:2] == [0, 0]
    assert 0.1 - TOLERANCE_S <= waits[2] <= 0.1


async def test_bucket_state_expires_once_it_would_be_full_again(redis: Redis, prefix: str) -> None:
    limiter = _limiter(redis, prefix, Bucket(rate_per_s=10, capacity=2))
    await limiter.acquire(ExternalApi.LLM)
    ttl_ms = await redis.pttl(f"{prefix}llm")
    assert 0 < ttl_ms <= 100 + 1000  # 1 token short of full at 10/s, plus the 1 s margin


async def test_the_script_is_reloaded_after_redis_forgets_it(redis: Redis, prefix: str) -> None:
    limiter = _limiter(redis, prefix, Bucket(rate_per_s=10, capacity=2))
    await limiter.acquire(ExternalApi.LLM)
    await redis.script_flush()  # what a Redis restart does to the script cache
    assert await limiter.acquire(ExternalApi.LLM) == 0


async def test_unreachable_redis_fails_closed() -> None:
    client = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.5)
    try:
        limiter = _limiter(client, "rl-test:", Bucket(rate_per_s=1, capacity=1))
        with pytest.raises(RetryableError) as caught:
            await limiter.acquire(ExternalApi.LLM)
        assert caught.value.code is ErrorCode.LIMITER_UNAVAILABLE
    finally:
        await client.aclose()
