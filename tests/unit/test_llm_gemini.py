"""Gemini provider: the real google-genai client against the fakes app (in-process) and against
scripted error responses. No network."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from common.config import LlmModelSpec, Settings
from common.errors import ErrorCode, ExternalApi, PermanentError, RetryableError
from fakes.app import app as fakes_app
from tests.unit.fakes import FakeLimiter, MemoryModelHealth
from worker.llm.base import LlmRequest, PromptImage, make_provider
from worker.llm.gemini import GeminiProvider
from worker.llm.prompts import synthesis_schema

FIRST, SECOND, THIRD = "gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"
CHAIN = (
    LlmModelSpec(model=FIRST, thinking="low", timeout_s=90),
    LlmModelSpec(model=SECOND, thinking="low", timeout_s=120),
    LlmModelSpec(model=THIRD, thinking="medium", timeout_s=60),
)
REQUEST = LlmRequest(
    system="rules",
    parts=("[0:00] intro\n[0:20] the slide\n[0:40] a diagram\n[1:00] summary",),
    schema=synthesis_schema(1, 2),
)

Responder = Callable[[httpx.Request], httpx.Response]


def _provider(
    transport: httpx.AsyncBaseTransport,
    limiter: FakeLimiter | None = None,
    *,
    health: MemoryModelHealth | None = None,
    step_down_on_overload: bool = False,
) -> GeminiProvider:
    return GeminiProvider(
        limiter or FakeLimiter(),
        api_key="test-key",
        chain=CHAIN,
        health=health or MemoryModelHealth(),
        step_down_on_overload=step_down_on_overload,
        base_url="http://llm.test",
        http_client=httpx.AsyncClient(transport=transport),
    )


class Scripted(httpx.AsyncBaseTransport):
    """Answers each request with the next responder and records the model each one asked for."""

    def __init__(self, *responders: Responder) -> None:
        self.responders = list(responders)
        self.models: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.models.append(request.url.path.rsplit("/", 1)[-1].split(":")[0])
        return self.responders.pop(0)(request)


def _ok(text: str = '{"title": "t"}') -> Responder:
    body = {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}
    return lambda request: httpx.Response(200, json=body)


def _error(
    code: int, status: str, message: str = "", details: list[Any] | None = None
) -> Responder:
    body = {"error": {"code": code, "status": status, "message": message, "details": details or []}}
    return lambda request: httpx.Response(code, json=body)


def _quota(quota_id: str, delay: str = "34s") -> list[dict[str, Any]]:
    return [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
                {"quotaMetric": "generate_content_free_tier_requests", "quotaId": quota_id}
            ],
        },
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
    ]


PER_MINUTE = _error(
    429,
    "RESOURCE_EXHAUSTED",
    "quota",
    _quota("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"),
)
PER_DAY = _error(
    429, "RESOURCE_EXHAUSTED", "quota", _quota("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
)


async def test_structured_request_round_trips_through_the_fake_gemini() -> None:
    sent: list[dict[str, Any]] = []

    async def record(request: httpx.Request) -> None:
        sent.append(json.loads(request.content))

    limiter = FakeLimiter()
    provider = GeminiProvider(
        limiter,
        api_key="test-key",
        chain=CHAIN,
        health=MemoryModelHealth(),
        base_url="http://fakes/llm",
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=fakes_app), event_hooks={"request": [record]}
        ),
    )
    notes = json.loads(await provider.generate_json(REQUEST))
    assert len(notes["frames"]) == 2  # the fake honours the schema's maxItems we sent
    config = sent[0]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"]["properties"]["frames"]["maxItems"] == 2
    assert config["responseJsonSchema"]["properties"]["frames"]["minItems"] == 1
    assert sent[0]["systemInstruction"]["parts"][0]["text"] == "rules"
    assert limiter.calls == [(ExternalApi.LLM, 1)]


async def test_candidate_frames_go_out_as_inline_images_at_the_configured_resolution() -> None:
    sent: list[dict[str, Any]] = []

    async def record(request: httpx.Request) -> None:
        sent.append(json.loads(request.content))

    provider = GeminiProvider(
        FakeLimiter(),
        api_key="test-key",
        chain=CHAIN,
        health=MemoryModelHealth(),
        image_resolution="medium",
        base_url="http://fakes/llm",
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=fakes_app), event_hooks={"request": [record]}
        ),
    )
    request = LlmRequest(
        system="rules",
        parts=(
            "[0:00] intro\n[0:20] Candidate frame t_sec=20:",
            PromptImage(data=b"\xff\xd8jpeg"),
            "\n[0:20] the slide",
        ),
        schema=synthesis_schema(1, 2),
    )
    notes = json.loads(await provider.generate_json(request))
    parts = sent[0]["contents"][0]["parts"]
    assert [sorted(p) for p in parts] == [["text"], ["inlineData", "mediaResolution"], ["text"]]
    assert parts[1]["inlineData"]["mime_type"] == "image/jpeg"  # as google-genai 2.33 sends it
    assert parts[1]["mediaResolution"] == {"level": "MEDIA_RESOLUTION_MEDIUM"}
    assert [f["t_sec"] for f in notes["frames"]] == [20]  # the fake picks among candidates


async def test_per_minute_429_is_retryable_with_googles_retry_delay_and_no_step_down() -> None:
    transport = Scripted(PER_MINUTE)
    health = MemoryModelHealth()
    with pytest.raises(RetryableError) as caught:
        await _provider(transport, health=health).generate_json(REQUEST)
    assert caught.value.code is ErrorCode.RATE_LIMITED
    assert caught.value.retry_after_s == 34
    assert caught.value.waiting_reason == "RATE_LIMITED:llm"
    assert transport.models == [FIRST]  # one HTTP call: the SDK's own retries are off
    assert health.bench == {}  # our own pacing, not the model's fault


async def test_daily_quota_benches_the_model_for_every_worker_and_steps_down() -> None:
    limiter = FakeLimiter()
    health = MemoryModelHealth(quota_reset_s=5000)
    transport = Scripted(PER_DAY, _ok(), _ok())
    provider = _provider(transport, limiter, health=health)
    assert await provider.generate_json(REQUEST) == '{"title": "t"}'
    await provider.generate_json(REQUEST)  # next chunk goes straight to the next model
    assert transport.models == [FIRST, SECOND, SECOND]
    assert len(limiter.calls) == 3  # every attempt takes a token
    assert health.bench == {FIRST: 5000}  # until the daily quota resets
    assert provider.answered_by == SECOND


async def test_a_model_benched_by_another_worker_is_skipped_without_a_call() -> None:
    limiter = FakeLimiter()
    health = MemoryModelHealth()
    health.bench[FIRST] = 300
    transport = Scripted(_ok())
    assert await _provider(transport, limiter, health=health).generate_json(REQUEST)
    assert transport.models == [SECOND]
    assert len(limiter.calls) == 1  # no token spent on the benched model


async def test_every_model_out_of_daily_quota_ends_as_a_normal_rate_limit() -> None:
    health = MemoryModelHealth()
    transport = Scripted(PER_DAY, PER_DAY, PER_DAY)
    with pytest.raises(RetryableError) as caught:
        await _provider(transport, health=health).generate_json(REQUEST)
    assert caught.value.code is ErrorCode.RATE_LIMITED
    assert transport.models == [FIRST, SECOND, THIRD]
    assert set(health.bench) == {FIRST, SECOND, THIRD}


async def test_with_every_model_benched_the_job_waits_for_the_first_one_back() -> None:
    limiter = FakeLimiter()
    health = MemoryModelHealth()
    health.bench.update({FIRST: 500, SECOND: 120, THIRD: 3600})
    transport = Scripted()
    with pytest.raises(RetryableError) as caught:
        await _provider(transport, limiter, health=health).generate_json(REQUEST)
    assert caught.value.code is ErrorCode.RATE_LIMITED
    assert caught.value.retry_after_s == 120
    assert transport.models == []
    assert limiter.calls == []


async def test_retry_delay_can_also_come_from_the_message() -> None:
    transport = Scripted(
        _error(429, "RESOURCE_EXHAUSTED", "Quota exceeded. Please retry in 12.5s.")
    )
    with pytest.raises(RetryableError) as caught:
        await _provider(transport).generate_json(REQUEST)
    assert caught.value.retry_after_s == 12.5


@pytest.mark.parametrize(
    ("code", "status", "expected", "retryable"),
    [
        (500, "INTERNAL", ErrorCode.UPSTREAM_ERROR, True),
        (503, "UNAVAILABLE", ErrorCode.UPSTREAM_ERROR, True),
        (504, "DEADLINE_EXCEEDED", ErrorCode.UPSTREAM_ERROR, True),
        (400, "INVALID_ARGUMENT", ErrorCode.UPSTREAM_REJECTED, False),
        (403, "PERMISSION_DENIED", ErrorCode.UPSTREAM_REJECTED, False),
        (404, "NOT_FOUND", ErrorCode.UPSTREAM_REJECTED, False),
    ],
)
async def test_other_http_errors_follow_the_shared_classification(
    code: int, status: str, expected: ErrorCode, retryable: bool
) -> None:
    transport = Scripted(_error(code, status, "nope"))
    with pytest.raises((RetryableError, PermanentError)) as caught:
        await _provider(transport).generate_json(REQUEST)
    assert (caught.value.code, caught.value.retryable) == (expected, retryable)
    assert len(transport.models) == 1


async def test_a_timeout_is_retryable() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(RetryableError) as caught:
        await _provider(Scripted(slow)).generate_json(REQUEST)
    assert caught.value.code is ErrorCode.UPSTREAM_TIMEOUT


async def test_an_answer_without_text_is_returned_empty_for_validation_to_reject() -> None:
    blocked = {"candidates": [{"finishReason": "SAFETY"}]}
    transport = Scripted(lambda request: httpx.Response(200, json=blocked))
    assert await _provider(transport).generate_json(REQUEST) == ""


HIGH_DEMAND = _error(503, "UNAVAILABLE", "This model is currently experiencing high demand.")


def _hang(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("no answer", request=request)


async def test_one_early_overload_waits_in_the_delay_queue_for_the_same_model() -> None:
    health = MemoryModelHealth()
    transport = Scripted(HIGH_DEMAND)
    with pytest.raises(RetryableError) as caught:
        await _provider(transport, health=health).generate_json(REQUEST)
    assert caught.value.code is ErrorCode.UPSTREAM_ERROR
    assert transport.models == [FIRST]  # a blip does not cost the stronger model
    assert health.strikes == {FIRST: 1}
    assert health.bench == {}


@pytest.mark.parametrize("failure", [HIGH_DEMAND, _hang], ids=["503", "timeout"])
async def test_a_repeated_overload_benches_the_model_and_steps_down_at_once(
    failure: Responder,
) -> None:
    health = MemoryModelHealth()
    health.strikes[FIRST] = 1  # another job, maybe on another worker, saw it fail already
    transport = Scripted(failure, _ok(), _ok())
    provider = _provider(transport, health=health)
    assert await provider.generate_json(REQUEST) == '{"title": "t"}'
    await provider.generate_json(REQUEST)  # the rest of this attempt stays on the next model
    assert transport.models == [FIRST, SECOND, SECOND]
    assert FIRST in health.bench


@pytest.mark.parametrize("failure", [HIGH_DEMAND, _hang], ids=["503", "timeout"])
async def test_later_attempts_step_down_on_the_first_overload(failure: Responder) -> None:
    limiter = FakeLimiter()
    health = MemoryModelHealth()
    transport = Scripted(failure, _ok())
    provider = _provider(transport, limiter, health=health, step_down_on_overload=True)
    assert await provider.generate_json(REQUEST) == '{"title": "t"}'
    assert transport.models == [FIRST, SECOND]
    assert len(limiter.calls) == 2
    assert health.bench == {}  # one strike: not benched for other jobs


async def test_an_answer_clears_the_models_strikes() -> None:
    health = MemoryModelHealth()
    health.strikes[FIRST] = 1
    await _provider(Scripted(_ok()), health=health).generate_json(REQUEST)
    assert health.strikes == {}


async def test_each_model_gets_its_own_thinking_level_and_timeout() -> None:
    sent: list[tuple[str, str, str]] = []  # model, thinking level, X-Server-Timeout

    def record(then: Responder) -> Responder:
        def respond(request: httpx.Request) -> httpx.Response:
            thinking = json.loads(request.content)["generationConfig"]["thinkingConfig"]
            level = thinking["thinking_level"]  # the SDK sends this one in snake_case
            model = request.url.path.rsplit("/", 1)[-1].split(":")[0]
            sent.append((model, level, request.headers["x-server-timeout"]))
            return then(request)

        return respond

    transport = Scripted(record(HIGH_DEMAND), record(_hang), record(_ok()))
    await _provider(transport, step_down_on_overload=True).generate_json(REQUEST)
    assert sent == [(FIRST, "LOW", "90"), (SECOND, "LOW", "120"), (THIRD, "MEDIUM", "60")]


async def test_a_rejected_request_never_steps_down() -> None:
    health = MemoryModelHealth()
    transport = Scripted(_error(400, "INVALID_ARGUMENT", "bad schema"))
    provider = _provider(transport, health=health, step_down_on_overload=True)
    with pytest.raises(PermanentError):
        await provider.generate_json(REQUEST)
    assert transport.models == [FIRST]
    assert health.strikes == {}
    assert health.bench == {}


async def test_an_overloaded_last_model_is_retried_later_as_usual() -> None:
    transport = Scripted(HIGH_DEMAND, HIGH_DEMAND, HIGH_DEMAND)
    with pytest.raises(RetryableError) as caught:
        await _provider(transport, step_down_on_overload=True).generate_json(REQUEST)
    assert caught.value.code is ErrorCode.UPSTREAM_ERROR
    assert transport.models == [FIRST, SECOND, THIRD]


@pytest.mark.parametrize(("attempt", "expected"), [(1, False), (2, False), (3, True), (5, True)])
def test_stepping_down_on_a_first_overload_starts_at_the_configured_attempt(
    attempt: int, expected: bool
) -> None:
    settings = _settings(fake_apis=True)
    provider = make_provider(settings, FakeLimiter(), MemoryModelHealth(), attempt)
    assert isinstance(provider, GeminiProvider)
    assert provider.step_down_on_overload is expected


def _settings(**values: Any) -> Settings:
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def test_the_provider_follows_the_configured_chain() -> None:
    chain = "gemini-3.5-flash-lite:high:45,gemini-3.6-flash:low:90"
    settings = _settings(fake_apis=True, llm_chain=chain)
    provider = make_provider(settings, FakeLimiter(), MemoryModelHealth())
    assert isinstance(provider, GeminiProvider)
    assert [str(spec) for spec in provider.chain] == chain.split(",")


def test_without_an_api_key_the_real_provider_is_refused() -> None:
    with pytest.raises(PermanentError) as caught:
        make_provider(_settings(), FakeLimiter(), MemoryModelHealth())
    assert caught.value.code is ErrorCode.UPSTREAM_REJECTED


def test_fake_apis_point_gemini_at_the_fakes_service_without_a_key() -> None:
    provider = make_provider(_settings(fake_apis=True), FakeLimiter(), MemoryModelHealth())
    assert isinstance(provider, GeminiProvider)


def test_providers_not_built_yet_say_so() -> None:
    with pytest.raises(NotImplementedError):
        make_provider(_settings(llm_provider="ollama"), FakeLimiter(), MemoryModelHealth())


async def test_the_fakes_service_can_make_a_model_answer_503_for_integration_tests() -> None:
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=fakes_app), base_url="http://fakes")
    await http.put("/_control/llm", json={"unavailable": [FIRST]})
    try:
        provider = GeminiProvider(
            FakeLimiter(),
            api_key="test-key",
            chain=CHAIN,
            health=MemoryModelHealth(),
            step_down_on_overload=True,
            base_url="http://fakes/llm",
            http_client=http,
        )
        assert json.loads(await provider.generate_json(REQUEST))["sections"]
        calls = (await http.get("/_control/llm/calls")).json()
        assert calls == [{"model": FIRST, "status": 503}, {"model": SECOND, "status": 200}]
    finally:
        await http.put("/_control/llm", json={})
