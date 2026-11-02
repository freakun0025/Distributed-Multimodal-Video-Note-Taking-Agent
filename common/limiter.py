"""Redis token bucket: acquire(api) before every external call; fail closed (ARCHITECTURE §9)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError

from common.config import Settings
from common.errors import ErrorCode, ExternalApi, RetryableError

KEY_PREFIX = "rl:"
_SCRIPT = Path(__file__).with_name("ratelimit.lua").read_text(encoding="utf-8")


@dataclass(frozen=True)
class Bucket:
    rate_per_s: float  # sustained calls per second
    capacity: float  # allowed burst; any window of w seconds admits capacity + rate * w calls


def buckets_from_settings(settings: Settings) -> dict[ExternalApi, Bucket]:
    """One bucket per external API, from the configured rates (all set below the real limits)."""
    return {
        ExternalApi.YOUTUBE: Bucket(settings.youtube_rpm / 60, settings.youtube_burst),
        ExternalApi.LLM: Bucket(settings.llm_rpm / 60, settings.llm_burst),
        ExternalApi.NOTION: Bucket(settings.notion_rps, settings.notion_burst),
    }


class Limiter(Protocol):
    async def acquire(self, api: ExternalApi, cost: float = 1) -> float: ...


class RateLimiter:
    """Shared by every worker through Redis, so the limit holds however many workers run."""

    def __init__(
        self,
        redis: Redis,
        buckets: Mapping[ExternalApi, Bucket],
        max_wait_s: float,
        *,
        key_prefix: str = KEY_PREFIX,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._script = redis.register_script(_SCRIPT)
        self._buckets = dict(buckets)
        self._max_wait_s = max_wait_s
        self._key_prefix = key_prefix
        self._sleep = sleep

    async def acquire(self, api: ExternalApi, cost: float = 1) -> float:
        """Take `cost` tokens before one call to `api`; returns the seconds waited.

        A short wait is booked in Redis and slept here. A wait longer than LIMITER_MAX_WAIT_S
        books nothing and raises RATE_LIMITED_LOCAL with retry_after_s, so the message goes to
        a delay queue instead of holding the worker. Redis unreachable → LIMITER_UNAVAILABLE.
        """
        bucket = self._buckets[api]
        if not 0 < cost <= bucket.capacity:
            raise ValueError(f"cost {cost} must be in (0, {bucket.capacity}] for {api.value}")
        try:
            reply = await self._script(
                keys=[f"{self._key_prefix}{api.value}"],
                args=[bucket.rate_per_s, bucket.capacity, cost, round(self._max_wait_s * 1000)],
            )
        except (RedisError, OSError) as exc:
            raise RetryableError(
                ErrorCode.LIMITER_UNAVAILABLE,
                f"rate limiter unavailable ({type(exc).__name__})",
                api=api,
            ) from exc
        wait_s = int(reply) / 1000
        if wait_s < 0:
            raise RetryableError(
                ErrorCode.RATE_LIMITED_LOCAL,
                f"{api.value} limiter wait {-wait_s:.1f}s exceeds {self._max_wait_s:g}s",
                api=api,
                retry_after_s=-wait_s,
            )
        if wait_s > 0:
            await self._sleep(wait_s)
        return wait_s
