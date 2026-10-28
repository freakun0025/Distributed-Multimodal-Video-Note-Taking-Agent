"""Error codes, Retryable/PermanentError and HTTP/exception classification (ARCHITECTURE §6.4)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import ClassVar

import httpx


class ExternalApi(StrEnum):
    """The three rate-limited external APIs; also the Redis bucket names (ARCHITECTURE §9)."""

    YOUTUBE = "youtube"
    LLM = "llm"
    NOTION = "notion"


class ErrorCode(StrEnum):
    # Retryable: delay queue while budget remains, then DLQ.
    RATE_LIMITED = "RATE_LIMITED"
    RATE_LIMITED_LOCAL = "RATE_LIMITED_LOCAL"
    LIMITER_UNAVAILABLE = "LIMITER_UNAVAILABLE"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    VIDEO_UNAVAILABLE = "VIDEO_UNAVAILABLE"
    STT_UNAVAILABLE = "STT_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    # Permanent: job -> FAILED.
    INVALID_URL = "INVALID_URL"
    MISSING_NOTION_TOKEN = "MISSING_NOTION_TOKEN"
    VIDEO_TOO_LONG = "VIDEO_TOO_LONG"
    VIDEO_UNSUPPORTED = "VIDEO_UNSUPPORTED"
    NO_TRANSCRIPT = "NO_TRANSCRIPT"
    NOTION_UNAUTHORIZED = "NOTION_UNAUTHORIZED"
    NOTION_PARENT_NOT_FOUND = "NOTION_PARENT_NOT_FOUND"
    LLM_INVALID_OUTPUT = "LLM_INVALID_OUTPUT"
    UPSTREAM_REJECTED = "UPSTREAM_REJECTED"
    TOKEN_DECRYPT_FAILED = "TOKEN_DECRYPT_FAILED"
    ENQUEUE_FAILED = "ENQUEUE_FAILED"
    OAUTH_CODE_INVALID = "OAUTH_CODE_INVALID"
    # REST API responses only; never a job error.
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_NOT_TERMINAL = "JOB_NOT_TERMINAL"
    BROKER_UNAVAILABLE = "BROKER_UNAVAILABLE"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class WarningCode(StrEnum):
    FRAME_EXTRACTION_FAILED = "FRAME_EXTRACTION_FAILED"


RETRYABLE_CODES: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.RATE_LIMITED,
        ErrorCode.RATE_LIMITED_LOCAL,
        ErrorCode.LIMITER_UNAVAILABLE,
        ErrorCode.UPSTREAM_ERROR,
        ErrorCode.UPSTREAM_TIMEOUT,
        ErrorCode.CONNECTION_ERROR,
        ErrorCode.VIDEO_UNAVAILABLE,
        ErrorCode.STT_UNAVAILABLE,
        ErrorCode.INTERNAL_ERROR,
    }
)

PERMANENT_CODES: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.INVALID_URL,
        ErrorCode.MISSING_NOTION_TOKEN,
        ErrorCode.VIDEO_TOO_LONG,
        ErrorCode.VIDEO_UNSUPPORTED,
        ErrorCode.NO_TRANSCRIPT,
        ErrorCode.NOTION_UNAUTHORIZED,
        ErrorCode.NOTION_PARENT_NOT_FOUND,
        ErrorCode.LLM_INVALID_OUTPUT,
        ErrorCode.UPSTREAM_REJECTED,
        ErrorCode.TOKEN_DECRYPT_FAILED,
        ErrorCode.ENQUEUE_FAILED,
        ErrorCode.OAUTH_CODE_INVALID,
    }
)

_DETAIL_MAX_CHARS = 200


class VnaError(Exception):
    """Base for pipeline errors. Stages raise a subclass; they never return error strings."""

    retryable: ClassVar[bool]
    allowed_codes: ClassVar[frozenset[ErrorCode]]

    def __init__(self, code: ErrorCode, message: str, *, api: ExternalApi | None = None) -> None:
        if code not in self.allowed_codes:
            raise ValueError(f"{code} is not a valid code for {type(self).__name__}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.api = api

    def to_info(self) -> dict[str, str]:
        """`{code, message}` as stored in `jobs.error` and returned by the API."""
        return {"code": self.code.value, "message": self.message}


class RetryableError(VnaError):
    retryable = True
    allowed_codes = RETRYABLE_CODES

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        api: ExternalApi | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(code, message, api=api)
        self.retry_after_s = retry_after_s

    @property
    def waiting_reason(self) -> str:
        """Value for `jobs.waiting_reason`, e.g. `RATE_LIMITED:llm`."""
        return f"{self.code.value}:{self.api.value}" if self.api else self.code.value


class PermanentError(VnaError):
    retryable = False
    allowed_codes = PERMANENT_CODES


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Seconds to wait from a `Retry-After` header (seconds or HTTP-date); None if absent/bad."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    try:
        return max(float(text), 0.0)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    return max((when - current).total_seconds(), 0.0)


def classify_http(
    api: ExternalApi,
    status_code: int,
    headers: Mapping[str, str] | None = None,
    *,
    detail: str | None = None,
    now: datetime | None = None,
) -> VnaError:
    """Map a failed HTTP response from an external API to the error the stage should raise."""
    if status_code < 400:
        raise ValueError(f"HTTP {status_code} is not an error")
    message = f"{api.value} returned HTTP {status_code}"
    if detail:
        message = f"{message}: {detail[:_DETAIL_MAX_CHARS]}"
    retry_after = parse_retry_after(_header(headers, "retry-after"), now=now)

    if status_code == 429:
        return RetryableError(ErrorCode.RATE_LIMITED, message, api=api, retry_after_s=retry_after)
    if status_code >= 500:
        return RetryableError(ErrorCode.UPSTREAM_ERROR, message, api=api, retry_after_s=retry_after)
    if status_code == 408:
        return RetryableError(ErrorCode.UPSTREAM_TIMEOUT, message, api=api)
    if api is ExternalApi.NOTION:
        if status_code in (401, 403):
            return PermanentError(ErrorCode.NOTION_UNAUTHORIZED, message, api=api)
        if status_code == 404:
            return PermanentError(ErrorCode.NOTION_PARENT_NOT_FOUND, message, api=api)
        if status_code == 409:  # Notion conflict_error: transient, safe to retry
            return RetryableError(ErrorCode.UPSTREAM_ERROR, message, api=api)
    if api is ExternalApi.YOUTUBE and status_code in (403, 404, 410):
        # Private/removed/blocked: ambiguous, so retryable -> DLQ after the budget (D12).
        return RetryableError(ErrorCode.VIDEO_UNAVAILABLE, message, api=api)
    return PermanentError(ErrorCode.UPSTREAM_REJECTED, message, api=api)


def classify_exception(api: ExternalApi, exc: BaseException) -> VnaError:
    """Map any exception from an external call to a VnaError; unknown -> INTERNAL_ERROR."""
    if isinstance(exc, VnaError):
        return exc
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        return classify_http(api, response.status_code, response.headers)
    if isinstance(exc, httpx.TimeoutException | TimeoutError):
        return RetryableError(ErrorCode.UPSTREAM_TIMEOUT, f"{api.value} timed out", api=api)
    if isinstance(exc, httpx.TransportError | ConnectionError):
        return RetryableError(ErrorCode.CONNECTION_ERROR, f"{api.value} connection failed", api=api)
    return RetryableError(
        ErrorCode.INTERNAL_ERROR, f"unexpected {type(exc).__name__} calling {api.value}", api=api
    )


def _header(headers: Mapping[str, str] | None, name: str) -> str | None:
    if headers is None:
        return None
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None
