"""Gemini provider via google-genai with structured JSON output (ARCHITECTURE §8.2).

generateContent (stateless, typed config) rather than the newer Interactions API, which stores
conversations server-side by default (D30). The SDK's own retries are switched off: retries go
through our delay queues, and every attempt takes a limiter token first (invariant 6).

Model chain (D36): each call goes to the first model of LLM_CHAIN that no worker has benched,
with that model's thinking level and timeout. It steps down to the next model, for the rest of
this attempt, when the model's daily quota is spent, when a 5xx/timeout benches it (repeated
failures, counted across workers), or on any 5xx/timeout once `step_down_on_overload` is set
(later attempts, D31). A single early 5xx/timeout instead retries the same model through the
delay queues, so one blip does not cost the stronger model.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from common.config import ImageResolution, LlmModelSpec
from common.errors import (
    ErrorCode,
    ExternalApi,
    RetryableError,
    VnaError,
    classify_exception,
    classify_http,
    parse_retry_after,
)
from common.limiter import Limiter
from common.logging import get_logger
from worker.llm.base import LlmRequest, PromptImage
from worker.llm.health import ModelHealth

API = ExternalApi.LLM
# Google's "high demand" 503s, other 5xx and hung calls: another model may still answer.
OVERLOAD_CODES = frozenset({ErrorCode.UPSTREAM_ERROR, ErrorCode.UPSTREAM_TIMEOUT})
_MESSAGE_MAX_CHARS = 200
_RETRY_IN = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_DURATION = re.compile(r"^(\d+(?:\.\d+)?)s$")
log = get_logger()


@dataclass(frozen=True)
class _Tuning:
    thinking: types.ThinkingConfig
    timeout_ms: int

    @classmethod
    def of(cls, spec: LlmModelSpec) -> _Tuning:
        return cls(
            thinking=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(spec.thinking.upper())
            ),
            timeout_ms=round(spec.timeout_s * 1000),
        )


class _ModelFailedError(Exception):
    """One model's attempt failed; `error` is what the stage sees if no step-down applies."""

    def __init__(self, error: VnaError, *, daily_quota: bool) -> None:
        super().__init__(str(error))
        self.error = error
        self.daily_quota = daily_quota


class GeminiProvider:
    def __init__(
        self,
        limiter: Limiter,
        *,
        api_key: str,
        chain: Sequence[LlmModelSpec],
        health: ModelHealth,
        image_resolution: ImageResolution = "medium",
        step_down_on_overload: bool = False,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not chain:
            raise ValueError("the model chain is empty")
        self._limiter = limiter
        self._chain = tuple(chain)
        self._health = health
        self._step_down_on_overload = step_down_on_overload
        self._current = 0  # index into the chain; only moves down, for the rest of this attempt
        self._answered_by: str | None = None
        self._image_resolution = types.PartMediaResolution(
            level=types.PartMediaResolutionLevel(f"MEDIA_RESOLUTION_{image_resolution.upper()}")
        )
        # Per model: thinking level and timeout (the SDK sends the timeout as X-Server-Timeout).
        self._tuning = {spec.model: _Tuning.of(spec) for spec in self._chain}
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                base_url=base_url,
                # milliseconds; every call overrides it with its model's own timeout
                timeout=max(tuning.timeout_ms for tuning in self._tuning.values()),
                retry_options=types.HttpRetryOptions(attempts=1),
                httpx_async_client=http_client,
            ),
        )

    @property
    def chain(self) -> tuple[LlmModelSpec, ...]:
        return self._chain

    @property
    def step_down_on_overload(self) -> bool:
        return self._step_down_on_overload

    @property
    def answered_by(self) -> str | None:
        """The model that gave the last answer (logged with the synthesis)."""
        return self._answered_by

    async def generate_json(self, request: LlmRequest) -> str:
        remaining = self._chain[self._current :]
        benched = await self._health.benched([spec.model for spec in remaining])
        failed: _ModelFailedError | None = None
        for index, spec in enumerate(remaining, start=self._current):
            if spec.model in benched:
                continue
            self._current = index
            try:
                text = await self._generate(spec, request)
            except _ModelFailedError as error:
                reason = await self._step_down_reason(spec, error)
                if reason is None:
                    raise error.error from error.__cause__
                failed = error
                log.warning(
                    "llm_step_down",
                    model=spec.model,
                    reason=reason,
                    code=error.error.code,
                    next=self._next_model(index, benched),
                )
                continue
            await self._health.answered(spec.model)
            self._answered_by = spec.model
            return text
        if failed is not None:  # the last model tried failed too: the delay queue retries
            raise failed.error from failed.__cause__
        back_in = min(benched.values())
        raise RetryableError(
            ErrorCode.RATE_LIMITED,
            f"every model in LLM_CHAIN is benched; the first is back in {back_in:.0f} s",
            api=API,
            retry_after_s=back_in,
        )

    async def _step_down_reason(self, spec: LlmModelSpec, failed: _ModelFailedError) -> str | None:
        """Why to try the next model now, or None to give the error to the delay queues."""
        if failed.daily_quota:
            back_in = await self._health.quota_spent(spec.model)
            log.warning("llm_model_benched", model=spec.model, reason="daily_quota", for_s=back_in)
            return "daily_quota"
        if failed.error.code not in OVERLOAD_CODES:
            return None  # rejected request, per-minute rate: another model would not help
        if await self._health.overloaded(spec.model):
            log.warning("llm_model_benched", model=spec.model, reason="overloaded")
            return "overloaded_repeatedly"
        return "overloaded" if self._step_down_on_overload else None

    def _next_model(self, index: int, benched: dict[str, float]) -> str | None:
        later = (spec.model for spec in self._chain[index + 1 :] if spec.model not in benched)
        return next(later, None)

    async def _generate(self, spec: LlmModelSpec, request: LlmRequest) -> str:
        await self._limiter.acquire(API)
        tuning = self._tuning[spec.model]
        config = types.GenerateContentConfig(
            system_instruction=request.system,
            response_mime_type="application/json",
            response_json_schema=dict(request.schema),
            thinking_config=tuning.thinking,
            http_options=types.HttpOptions(timeout=tuning.timeout_ms),
            # No tools, and no SDK-driven follow-up calls that would bypass the limiter.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        model = spec.model
        try:
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=self._parts(request))],
                config=config,
            )
        except errors.APIError as exc:
            error = classify_gemini_error(exc)
            raise _ModelFailedError(error, daily_quota=_is_daily_quota(exc)) from exc
        except Exception as exc:  # timeouts, connection errors from httpx
            raise _ModelFailedError(classify_exception(API, exc), daily_quota=False) from exc
        usage = response.usage_metadata
        if usage is not None:  # measured, not estimated (CP7b)
            log.info(
                "llm_usage",
                model=model,
                images=len(request.images),
                prompt_tokens=usage.prompt_token_count,
                output_tokens=usage.candidates_token_count,
                thought_tokens=usage.thoughts_token_count,
            )
        return response.text or ""

    def _parts(self, request: LlmRequest) -> list[types.Part]:
        """Text and images in reading order; every image at the configured resolution."""
        return [
            types.Part(
                inline_data=types.Blob(data=part.data, mime_type=part.mime_type),
                media_resolution=self._image_resolution,
            )
            if isinstance(part, PromptImage)
            else types.Part.from_text(text=part)
            for part in request.parts
        ]


def classify_gemini_error(exc: errors.APIError) -> VnaError:
    message = f"{exc.status or exc.code}: {exc.message or ''}".strip()[:_MESSAGE_MAX_CHARS]
    if exc.code == 429:
        return RetryableError(
            ErrorCode.RATE_LIMITED, message, api=API, retry_after_s=_retry_after(exc)
        )
    return classify_http(API, exc.code, detail=message)


def _is_daily_quota(exc: errors.APIError) -> bool:
    """429 because the per-day quota is spent (vs. the per-minute rate): switch models."""
    if exc.code != 429:
        return False
    texts = [str(exc.message or ""), *_strings(exc.details)]
    return any(
        "perday" in text.replace("_", "").replace("-", "").lower()
        or "quota_exceeded" in text
        or "per day" in text.lower()
        for text in texts
    )


def _retry_after(exc: errors.APIError) -> float | None:
    """RetryInfo.retryDelay ("34s"), "retry in 34.5s" in the message, or Retry-After."""
    for node in _dicts(exc.details):
        delay = node.get("retryDelay")
        if isinstance(delay, str) and (match := _DURATION.match(delay.strip())):
            return float(match.group(1))
    if match := _RETRY_IN.search(str(exc.message or "")):
        return float(match.group(1))
    headers = getattr(exc.response, "headers", None)
    return parse_retry_after(headers.get("retry-after")) if headers is not None else None


def _dicts(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _dicts(value)
    elif isinstance(node, list):
        for item in node:
            yield from _dicts(item)


def _strings(node: Any) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)
