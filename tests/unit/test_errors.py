from datetime import UTC, datetime

import httpx
import pytest

from common.errors import (
    PERMANENT_CODES,
    RETRYABLE_CODES,
    ErrorCode,
    ExternalApi,
    PermanentError,
    RetryableError,
    classify_exception,
    classify_http,
    parse_retry_after,
)

NOW = datetime(2026, 10, 28, 12, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize("api", list(ExternalApi))
def test_429_from_any_api_is_retryable_rate_limited(api: ExternalApi) -> None:
    error = classify_http(api, 429, {"Retry-After": "12"})
    assert isinstance(error, RetryableError)
    assert error.code is ErrorCode.RATE_LIMITED
    assert error.retry_after_s == 12
    assert error.waiting_reason == f"RATE_LIMITED:{api.value}"


@pytest.mark.parametrize("status", [500, 502, 503, 504])
@pytest.mark.parametrize("api", list(ExternalApi))
def test_5xx_is_retryable(api: ExternalApi, status: int) -> None:
    error = classify_http(api, status)
    assert isinstance(error, RetryableError)
    assert error.code is ErrorCode.UPSTREAM_ERROR


def test_408_is_a_retryable_timeout() -> None:
    error = classify_http(ExternalApi.LLM, 408)
    assert isinstance(error, RetryableError)
    assert error.code is ErrorCode.UPSTREAM_TIMEOUT


@pytest.mark.parametrize("status", [401, 403])
def test_notion_auth_failures_are_permanent_unauthorized(status: int) -> None:
    error = classify_http(ExternalApi.NOTION, status)
    assert isinstance(error, PermanentError)
    assert error.code is ErrorCode.NOTION_UNAUTHORIZED


def test_notion_404_means_parent_not_found_or_not_shared() -> None:
    error = classify_http(ExternalApi.NOTION, 404)
    assert isinstance(error, PermanentError)
    assert error.code is ErrorCode.NOTION_PARENT_NOT_FOUND


def test_notion_409_conflict_is_retryable() -> None:
    assert isinstance(classify_http(ExternalApi.NOTION, 409), RetryableError)


@pytest.mark.parametrize("status", [403, 404, 410])
def test_youtube_unavailable_is_retryable_so_it_reaches_the_dlq(status: int) -> None:
    error = classify_http(ExternalApi.YOUTUBE, status)
    assert isinstance(error, RetryableError)
    assert error.code is ErrorCode.VIDEO_UNAVAILABLE


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_other_llm_4xx_are_permanent_rejections(status: int) -> None:
    error = classify_http(ExternalApi.LLM, status)
    assert isinstance(error, PermanentError)
    assert error.code is ErrorCode.UPSTREAM_REJECTED


def test_classify_http_refuses_success_codes() -> None:
    with pytest.raises(ValueError):
        classify_http(ExternalApi.LLM, 200)


def test_long_error_detail_is_truncated() -> None:
    error = classify_http(ExternalApi.NOTION, 400, detail="x" * 5000)
    assert len(error.message) < 300


def test_retry_after_header_lookup_is_case_insensitive() -> None:
    error = classify_http(ExternalApi.NOTION, 429, {"retry-after": "3"})
    assert isinstance(error, RetryableError)
    assert error.retry_after_s == 3


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("30", 30.0),
        ("1.5", 1.5),
        ("-4", 0.0),
        ("Wed, 28 Oct 2026 12:01:30 GMT", 90.0),
        ("Wed, 28 Oct 2026 11:00:00 GMT", 0.0),  # in the past
        ("soon", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_retry_after_accepts_seconds_and_http_dates(
    value: str | None, expected: float | None
) -> None:
    assert parse_retry_after(value, now=NOW) == expected


def test_timeouts_and_connection_errors_are_retryable() -> None:
    request = httpx.Request("GET", "https://api.notion.com/v1/pages")
    timeout = classify_exception(ExternalApi.NOTION, httpx.ReadTimeout("t", request=request))
    connect = classify_exception(ExternalApi.NOTION, httpx.ConnectError("c", request=request))
    assert timeout.code is ErrorCode.UPSTREAM_TIMEOUT
    assert connect.code is ErrorCode.CONNECTION_ERROR
    assert classify_exception(ExternalApi.LLM, TimeoutError()).code is ErrorCode.UPSTREAM_TIMEOUT


def test_http_status_error_is_classified_by_its_response() -> None:
    request = httpx.Request("POST", "https://api.notion.com/v1/pages")
    response = httpx.Response(429, headers={"Retry-After": "7"}, request=request)
    exc = httpx.HTTPStatusError("429", request=request, response=response)
    error = classify_exception(ExternalApi.NOTION, exc)
    assert isinstance(error, RetryableError)
    assert error.retry_after_s == 7


def test_unknown_exceptions_are_retryable_internal_errors() -> None:
    error = classify_exception(ExternalApi.LLM, KeyError("boom"))
    assert isinstance(error, RetryableError)
    assert error.code is ErrorCode.INTERNAL_ERROR


def test_existing_pipeline_errors_pass_through_unchanged() -> None:
    original = PermanentError(ErrorCode.VIDEO_TOO_LONG, "too long")
    assert classify_exception(ExternalApi.YOUTUBE, original) is original


def test_error_classes_only_accept_codes_of_their_kind() -> None:
    with pytest.raises(ValueError):
        RetryableError(ErrorCode.VIDEO_TOO_LONG, "x")
    with pytest.raises(ValueError):
        PermanentError(ErrorCode.RATE_LIMITED, "x")


def test_retryable_and_permanent_code_sets_do_not_overlap() -> None:
    assert not RETRYABLE_CODES & PERMANENT_CODES


def test_to_info_is_the_api_error_shape() -> None:
    error = PermanentError(ErrorCode.VIDEO_TOO_LONG, "Video is longer than 20 minutes.")
    assert error.to_info() == {
        "code": "VIDEO_TOO_LONG",
        "message": "Video is longer than 20 minutes.",
    }
