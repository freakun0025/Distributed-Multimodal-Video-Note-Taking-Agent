"""Gemini provider via google-genai with structured JSON output (ARCHITECTURE §8.2).

generateContent (stateless, typed config) rather than the newer Interactions API, which stores
conversations server-side by default (D30). The SDK's own retries are switched off: retries go
through our delay queues, and every attempt takes a limiter token first (invariant 6).

Fallback model, for the rest of the job: when the primary's daily quota is spent, or - only
once `fallback_on_overload` is set (later attempts, D31) - when it is overloaded (5xx/timeout).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from common.config import ImageResolution, ThinkingLevel
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
    def of(cls, level: ThinkingLevel, timeout_s: float) -> _Tuning:
        return cls(
            thinking=types.ThinkingConfig(thinking_level=types.ThinkingLevel(level.upper())),
            timeout_ms=round(timeout_s * 1000),
        )


class _ModelFailedError(Exception):
    """One model's attempt failed; `error` is what the stage sees if no fallback applies."""

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
        model: str,
        fallback_model: str | None,
        thinking_level: ThinkingLevel,
        timeout_s: float,
        fallback_thinking_level: ThinkingLevel | None = None,
        fallback_timeout_s: float | None = None,
        image_resolution: ImageResolution = "medium",
        fallback_on_overload: bool = False,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._limiter = limiter
        self._model = model
        self._fallback_model = fallback_model
        self._fallback_on_overload = fallback_on_overload
        self._on_fallback = False  # once switched, the rest of this job uses the fallback
        self._image_resolution = types.PartMediaResolution(
            level=types.PartMediaResolutionLevel(f"MEDIA_RESOLUTION_{image_resolution.upper()}")
        )
        # Per model: thinking level and timeout (the SDK sends the timeout as X-Server-Timeout).
        self._tuning = {
            False: _Tuning.of(thinking_level, timeout_s),
            True: _Tuning.of(
                fallback_thinking_level or thinking_level, fallback_timeout_s or timeout_s
            ),
        }
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                base_url=base_url,
                timeout=round(timeout_s * 1000),  # milliseconds; overridden per call
                retry_options=types.HttpRetryOptions(attempts=1),
                httpx_async_client=http_client,
            ),
        )

    @property
    def model(self) -> str:
        return self._fallback_model if self._on_fallback and self._fallback_model else self._model

    @property
    def fallback_on_overload(self) -> bool:
        return self._fallback_on_overload

    async def generate_json(self, request: LlmRequest) -> str:
        try:
            return await self._generate(self.model, request)
        except _ModelFailedError as failed:
            reason = self._fallback_reason(failed)
            if reason is None:
                raise failed.error from failed.__cause__
            log.warning(
                "llm_fallback",
                reason=reason,
                model=self._model,
                fallback=self._fallback_model,
                code=failed.error.code,
            )
        self._on_fallback = True
        try:
            return await self._generate(self.model, request)
        except _ModelFailedError as failed_again:
            raise failed_again.error from failed_again.__cause__

    def _fallback_reason(self, failed: _ModelFailedError) -> str | None:
        if self._on_fallback or not self._fallback_model:
            return None
        if failed.daily_quota:
            return "daily_quota"
        if self._fallback_on_overload and failed.error.code in OVERLOAD_CODES:
            return "overloaded"
        return None

    async def _generate(self, model: str, request: LlmRequest) -> str:
        await self._limiter.acquire(API)
        tuning = self._tuning[self._on_fallback]
        config = types.GenerateContentConfig(
            system_instruction=request.system,
            response_mime_type="application/json",
            response_json_schema=dict(request.schema),
            thinking_config=tuning.thinking,
            http_options=types.HttpOptions(timeout=tuning.timeout_ms),
            # No tools, and no SDK-driven follow-up calls that would bypass the limiter.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
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
