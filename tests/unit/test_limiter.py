"""RateLimiter decisions around the Lua reply (ARCHITECTURE §9); the Lua maths runs against real
Redis in tests/integration/test_limiter.py."""

from collections.abc import Sequence
from typing import Any

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from common.config import Settings
from common.errors import ErrorCode, ExternalApi, RetryableError
from common.limiter import Bucket, RateLimiter, buckets_from_settings

# gemini-3.8-flash free tier for our project, read in AI Studio (ARCHITECTURE §9).
MEASURED_LLM_RPM = 5
BUCKETS = {ExternalApi.LLM: Bucket(rate_per_s=4 / 60, capacity=1)}


class FakeScript:
    def __init__(self, reply: int | Exception) -> None:
        self.reply = reply
        self.calls: list[tuple[list[str], list[Any]]] = []

    async def __call__(
        self, keys: Sequence[str] | None = None, args: Sequence[Any] | None = None
    ) -> int:
        self.calls.append((list(keys or []), list(args or [])))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class FakeRedis:
    def __init__(self, reply: int | Exception) -> None:
        self.script = FakeScript(reply)
        self.registered: list[str] = []

    def register_script(self, script: str) -> FakeScript:
        self.registered.append(script)
        return self.script


def _limiter(reply: int | Exception, sleeps: list[float]) -> tuple[RateLimiter, FakeRedis]:
    redis = FakeRedis(reply)

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    limiter = RateLimiter(redis, BUCKETS, max_wait_s=20, sleep=sleep)  # type: ignore[arg-type]
    return limiter, redis


async def test_a_free_token_is_taken_without_waiting() -> None:
    sleeps: list[float] = []
    limiter, redis = _limiter(0, sleeps)
    assert await limiter.acquire(ExternalApi.LLM) == 0
    assert sleeps == []
    assert redis.script.calls == [(["rl:llm"], [4 / 60, 1, 1, 20000])]
    assert "TIME" in redis.registered[0]  # Redis clock, not the worker's


async def test_a_short_wait_is_booked_and_slept_here() -> None:
    sleeps: list[float] = []
    limiter, _ = _limiter(1500, sleeps)
    assert await limiter.acquire(ExternalApi.LLM) == 1.5
    assert sleeps == [1.5]


async def test_a_wait_past_the_limit_becomes_a_delayed_retry_instead_of_holding_the_worker() -> (
    None
):
    sleeps: list[float] = []
    limiter, _ = _limiter(-45000, sleeps)
    with pytest.raises(RetryableError) as caught:
        await limiter.acquire(ExternalApi.LLM)
    assert caught.value.code is ErrorCode.RATE_LIMITED_LOCAL
    assert caught.value.retry_after_s == 45
    assert caught.value.api is ExternalApi.LLM
    assert caught.value.waiting_reason == "RATE_LIMITED_LOCAL:llm"
    assert sleeps == []


@pytest.mark.parametrize(
    "failure",
    [RedisConnectionError("refused"), RedisTimeoutError("slow"), OSError("network down")],
)
async def test_redis_failure_fails_closed(failure: Exception) -> None:
    limiter, _ = _limiter(failure, [])
    with pytest.raises(RetryableError) as caught:
        await limiter.acquire(ExternalApi.LLM)
    assert caught.value.code is ErrorCode.LIMITER_UNAVAILABLE


@pytest.mark.parametrize("cost", [0, -1, 2])
async def test_a_cost_the_bucket_could_never_grant_is_a_programming_error(cost: float) -> None:
    limiter, redis = _limiter(0, [])
    with pytest.raises(ValueError, match="cost"):
        await limiter.acquire(ExternalApi.LLM, cost=cost)
    assert redis.script.calls == []


def test_buckets_come_from_the_configured_rates() -> None:
    buckets = buckets_from_settings(Settings(_env_file=None))  # type: ignore[call-arg]
    assert buckets == {
        ExternalApi.YOUTUBE: Bucket(rate_per_s=0.5, capacity=2),
        ExternalApi.LLM: Bucket(rate_per_s=4 / 60, capacity=1),
        ExternalApi.NOTION: Bucket(rate_per_s=2.5, capacity=3),
    }


def test_default_llm_bucket_admits_no_more_than_the_measured_rpm_in_any_minute() -> None:
    llm = buckets_from_settings(Settings(_env_file=None))[ExternalApi.LLM]  # type: ignore[call-arg]
    assert llm.capacity + llm.rate_per_s * 60 <= MEASURED_LLM_RPM
