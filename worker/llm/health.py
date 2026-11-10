"""Model health shared by every worker through Redis (ARCHITECTURE §8.2, D36).

A model is *benched* (skipped by every worker) when its daily quota is spent, until the quota
resets, or after LLM_MODEL_STRIKES 5xx/timeouts within LLM_STRIKE_WINDOW_S, for
LLM_MODEL_COOLDOWN_S. One job discovering an outage spares every later job the wait. Benches
expire on their own (Redis TTL); an answer clears the model's strikes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, tzinfo
from typing import Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError

from common.config import Settings
from common.errors import ErrorCode, ExternalApi, RetryableError

KEY_PREFIX = "llm:"

# Count a strike; at the threshold bench the model and start counting afresh. Atomic, so two
# workers failing at once cannot both miss the threshold.
_STRIKE = """
local strikes = redis.call('INCR', KEYS[1])
if strikes == 1 then redis.call('PEXPIRE', KEYS[1], ARGV[2]) end
if strikes >= tonumber(ARGV[1]) then
  redis.call('SET', KEYS[2], 'overloaded', 'PX', ARGV[3])
  redis.call('DEL', KEYS[1])
  return 1
end
return 0
"""


class ModelHealth(Protocol):
    async def benched(self, models: Sequence[str]) -> dict[str, float]:
        """The benched ones among `models`: model -> seconds until it may be used again."""
        ...

    async def quota_spent(self, model: str) -> float:
        """Bench `model` until its daily quota resets; returns the seconds benched."""
        ...

    async def overloaded(self, model: str) -> bool:
        """Count a 5xx/timeout; True if that benched the model."""
        ...

    async def answered(self, model: str) -> None:
        """The model answered: forget its strikes."""
        ...


def seconds_until_midnight(zone: tzinfo, now: datetime | None = None) -> float:
    """Seconds from `now` until the next midnight in `zone` (when Gemini's RPD resets)."""
    current = (now or datetime.now(zone)).astimezone(zone)
    midnight = datetime.combine(current.date() + timedelta(days=1), datetime.min.time(), zone)
    # Timestamps, not datetime subtraction: same-tzinfo subtraction ignores a DST change.
    return max(midnight.timestamp() - current.timestamp(), 1.0)


class RedisModelHealth:
    def __init__(
        self,
        redis: Redis,
        *,
        strikes: int,
        strike_window_s: float,
        cooldown_s: float,
        quota_reset_s: Callable[[], float],
        key_prefix: str = KEY_PREFIX,
    ) -> None:
        self._redis = redis
        self._strike = redis.register_script(_STRIKE)
        self._strikes = strikes
        self._window_ms = round(strike_window_s * 1000)
        self._cooldown_ms = round(cooldown_s * 1000)
        self._quota_reset_s = quota_reset_s
        self._prefix = key_prefix

    @classmethod
    def from_settings(cls, redis: Redis, settings: Settings, zone: tzinfo) -> RedisModelHealth:
        return cls(
            redis,
            strikes=settings.llm_model_strikes,
            strike_window_s=settings.llm_strike_window_s,
            cooldown_s=settings.llm_model_cooldown_s,
            quota_reset_s=lambda: seconds_until_midnight(zone),
        )

    def bench_key(self, model: str) -> str:
        return f"{self._prefix}bench:{model}"

    def strikes_key(self, model: str) -> str:
        return f"{self._prefix}strikes:{model}"

    async def benched(self, models: Sequence[str]) -> dict[str, float]:
        if not models:
            return {}
        try:
            async with self._redis.pipeline(transaction=False) as pipe:
                for model in models:
                    pipe.pttl(self.bench_key(model))
                remaining = await pipe.execute()
        except (RedisError, OSError) as exc:
            raise _unavailable(exc) from exc
        return {
            model: int(ms) / 1000 for model, ms in zip(models, remaining, strict=True) if ms > 0
        }

    async def quota_spent(self, model: str) -> float:
        seconds = self._quota_reset_s()
        try:
            await self._redis.set(self.bench_key(model), "daily_quota", px=round(seconds * 1000))
        except (RedisError, OSError) as exc:
            raise _unavailable(exc) from exc
        return seconds

    async def overloaded(self, model: str) -> bool:
        try:
            benched = await self._strike(
                keys=[self.strikes_key(model), self.bench_key(model)],
                args=[self._strikes, self._window_ms, self._cooldown_ms],
            )
        except (RedisError, OSError) as exc:
            raise _unavailable(exc) from exc
        return bool(int(benched))

    async def answered(self, model: str) -> None:
        try:
            await self._redis.delete(self.strikes_key(model))
        except (RedisError, OSError) as exc:
            raise _unavailable(exc) from exc


def _unavailable(exc: Exception) -> RetryableError:
    """Same Redis as the limiter, so the same code: fail closed, retry later."""
    return RetryableError(
        ErrorCode.LIMITER_UNAVAILABLE,
        f"model health unavailable ({type(exc).__name__})",
        api=ExternalApi.LLM,
    )
