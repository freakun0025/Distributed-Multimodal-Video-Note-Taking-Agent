"""The fakes' failure switches (CP9, Spec §6 on demand), driven through the worker's real clients
in-process, so each injected failure is classified exactly as the real API's would be."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

import fakes.app
from common.errors import ErrorCode, ExternalApi, RetryableError
from tests.unit.fakes import VIDEO_ID, FakeLimiter, MemoryModelHealth
from tests.unit.test_llm_gemini import CHAIN, FIRST, REQUEST
from worker.clients.fakes import FakesYouTubeClient
from worker.clients.notion import NotionClient
from worker.llm.gemini import GeminiProvider


@pytest.fixture
async def fakes_http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fakes.app.app), base_url="http://fakes"
    ) as http:
        await http.post("/_control/reset")
        yield http
        await http.post("/_control/reset")


def _youtube(http: httpx.AsyncClient) -> FakesYouTubeClient:
    return FakesYouTubeClient(FakeLimiter(), "http://fakes", 5, http_client=http)


async def _statuses(http: httpx.AsyncClient, api: str) -> list[int]:
    calls = (await http.get("/_control/calls", params={"api": api})).json()
    return [call["status"] for call in calls]


async def test_a_fault_answers_the_next_calls_then_the_api_works_again(
    fakes_http: httpx.AsyncClient,
) -> None:
    switch = {"api": "youtube", "count": 2, "retry_after": 20}
    assert (await fakes_http.put("/_control/faults", json=switch)).status_code == 200
    youtube = _youtube(fakes_http)
    for _ in range(2):
        with pytest.raises(RetryableError) as caught:
            await youtube.video_info(VIDEO_ID)
        assert caught.value.code is ErrorCode.RATE_LIMITED
        assert caught.value.retry_after_s == 20
        assert caught.value.waiting_reason == "RATE_LIMITED:youtube"
    assert (await youtube.video_info(VIDEO_ID)).duration_sec == fakes.app.DURATION_S
    assert await _statuses(fakes_http, "youtube") == [429, 429, 200]
    assert (await fakes_http.get("/_control/switches")).json()["faults"] == {}


async def test_a_fake_llm_429_is_a_per_minute_limit_with_googles_retry_delay(
    fakes_http: httpx.AsyncClient,
) -> None:
    await fakes_http.put("/_control/faults", json={"api": "llm", "retry_after": 25})
    health = MemoryModelHealth()
    provider = GeminiProvider(
        FakeLimiter(), api_key="test-key", chain=CHAIN, health=health,
        base_url="http://fakes/llm", http_client=fakes_http,
    )  # fmt: skip
    with pytest.raises(RetryableError) as caught:
        await provider.generate_json(REQUEST)
    assert caught.value.code is ErrorCode.RATE_LIMITED
    assert caught.value.retry_after_s == 25  # RetryInfo.retryDelay in Google's error body
    assert health.bench == {}  # per-minute, not the daily quota: no model is benched (D36)
    assert await provider.generate_json(REQUEST)  # the next call works, on the same model
    assert provider.answered_by == FIRST


def _gemini(http: httpx.AsyncClient, health: MemoryModelHealth) -> GeminiProvider:
    return GeminiProvider(
        FakeLimiter(), api_key="test-key", chain=CHAIN, health=health,
        base_url="http://fakes/llm", http_client=http,
    )  # fmt: skip


async def _llm_statuses(http: httpx.AsyncClient) -> list[int]:
    return [call["status"] for call in (await http.get("/_control/llm/calls")).json()]


async def test_the_fake_llm_answers_429_past_its_calls_per_minute_like_gemini(
    fakes_http: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    monkeypatch.setattr(fakes.app, "_monotonic", lambda: now[0])
    assert (await fakes_http.put("/_control/llm", json={"rpm": 2})).json()["rpm"] == 2
    health = MemoryModelHealth()
    provider = _gemini(fakes_http, health)
    assert await provider.generate_json(REQUEST)
    now[0] += 20
    assert await provider.generate_json(REQUEST)
    now[0] += 20
    with pytest.raises(RetryableError) as caught:
        await provider.generate_json(REQUEST)
    assert caught.value.code is ErrorCode.RATE_LIMITED
    assert caught.value.retry_after_s == 20  # until the first call leaves the minute
    assert health.bench == {}  # per-minute, never taken for the daily quota (D36)
    now[0] += 20  # the first call is a minute old: room for one more
    assert await provider.generate_json(REQUEST)
    assert await _llm_statuses(fakes_http) == [200, 200, 429, 200]


async def test_the_default_fake_llm_limit_is_the_free_tier_rpm_and_0_turns_it_off(
    fakes_http: httpx.AsyncClient,
) -> None:
    switches = (await fakes_http.get("/_control/switches")).json()
    assert switches["llm_rpm"] == fakes.app.FAKE_LLM_RPM == 5
    await fakes_http.put("/_control/llm", json={"rpm": 0})
    provider = _gemini(fakes_http, MemoryModelHealth())
    for _ in range(7):
        assert await provider.generate_json(REQUEST)
    assert set(await _llm_statuses(fakes_http)) == {200}
    for bad in (-1, "5", 2.5, True):
        assert (await fakes_http.put("/_control/llm", json={"rpm": bad})).status_code == 422


async def test_a_fake_notion_429_carries_retry_after(fakes_http: httpx.AsyncClient) -> None:
    await fakes_http.put("/_control/faults", json={"api": "notion", "retry_after": 3})
    notion = NotionClient(
        "token", base_url="http://fakes/notion", version="2026-03-11", limiter=FakeLimiter(),
        timeout_s=5, http_client=fakes_http,
    )  # fmt: skip
    async with notion:
        with pytest.raises(RetryableError) as caught:
            await notion.create_page("parent", "Fake · #12345678")
        assert caught.value.code is ErrorCode.RATE_LIMITED
        assert caught.value.retry_after_s == 3
        assert "rate_limited" in str(caught.value)
        assert (await notion.create_page("parent", "Fake · #12345678")).id


async def test_a_5xx_switch_is_an_upstream_error(fakes_http: httpx.AsyncClient) -> None:
    await fakes_http.put("/_control/faults", json={"api": "youtube", "status": 503})
    with pytest.raises(RetryableError) as caught:
        await _youtube(fakes_http).video_info(VIDEO_ID)
    assert caught.value.code is ErrorCode.UPSTREAM_ERROR


async def test_the_oauth_exchange_is_not_a_pipeline_call(fakes_http: httpx.AsyncClient) -> None:
    await fakes_http.put("/_control/faults", json={"api": "notion"})
    response = await fakes_http.post(
        "/notion/v1/oauth/token",
        json={"grant_type": "authorization_code", "code": "c"},
        headers={"authorization": "Basic eA=="},
    )
    assert response.status_code == 200  # the gateway's exchange never gets the fault
    assert (await fakes_http.get("/_control/switches")).json()["faults"]["notion"]["remaining"] == 1


async def test_a_private_video_is_unavailable_and_retryable(
    fakes_http: httpx.AsyncClient, tmp_path: Path
) -> None:
    await fakes_http.put("/_control/youtube", json={"unavailable": True})
    youtube = _youtube(fakes_http)
    with pytest.raises(RetryableError) as caught:
        await youtube.video_info(VIDEO_ID)
    assert caught.value.code is ErrorCode.VIDEO_UNAVAILABLE  # D12: retried, then the DLQ
    assert caught.value.api is ExternalApi.YOUTUBE
    assert "Private video" in str(caught.value)

    async def progress(_: float) -> None:
        pass

    with pytest.raises(RetryableError) as download:
        await youtube.download_video(
            VIDEO_ID, tmp_path, name="video", max_height=720, progress=progress
        )
    assert download.value.code is ErrorCode.VIDEO_UNAVAILABLE
    assert not await asyncio.to_thread(lambda: list(tmp_path.iterdir()))  # no partial file left


@pytest.fixture
def videos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Distinct bytes for the intact, cut-short and black videos."""
    content = {"ok": b"intact" * 100, "partial": b"cut short" * 50, "none": b"black" * 120}
    for mode, data in content.items():
        (tmp_path / f"{mode}.mp4").write_bytes(data)
    monkeypatch.setattr(fakes.app, "VIDEO_PATH", tmp_path / "ok.mp4")
    monkeypatch.setattr(
        fakes.app, "FRAME_FAULT_VIDEOS", {m: tmp_path / f"{m}.mp4" for m in ("partial", "none")}
    )
    monkeypatch.setattr(fakes.app, "VIDEO_STREAM_SECONDS", 0.0)
    return content


@pytest.mark.parametrize("mode", ["partial", "none"])
async def test_frame_faults_break_only_the_frames_download(
    fakes_http: httpx.AsyncClient, videos: dict[str, bytes], tmp_path: Path, mode: str
) -> None:
    await fakes_http.put("/_control/youtube", json={"frames": mode})
    youtube = _youtube(fakes_http)

    async def progress(_: float) -> None:
        pass

    preview = await youtube.download_video(
        VIDEO_ID, tmp_path / "job", name="preview", max_height=480, progress=progress
    )
    frames = await youtube.download_video(
        VIDEO_ID, tmp_path / "job", name="video", max_height=720, progress=progress
    )
    # The notes still see the whole video; only EXTRACT_FRAMES gets the broken one.
    assert preview.read_bytes() == videos["ok"]
    assert frames.read_bytes() == videos[mode]


async def test_reset_makes_every_api_work_again(fakes_http: httpx.AsyncClient) -> None:
    await fakes_http.put("/_control/faults", json={"api": "llm", "count": 3})
    await fakes_http.put("/_control/youtube", json={"unavailable": True, "frames": "none"})
    await fakes_http.put("/_control/llm", json={"unavailable": [FIRST]})
    switches = (await fakes_http.get("/_control/switches")).json()
    assert switches["faults"] == {"llm": {"status": 429, "remaining": 3, "retry_after": None}}
    assert switches["youtube"] == {"unavailable": True, "frames": "none", "duration": 600}
    assert switches["llm_unavailable"] == [FIRST]

    assert (await fakes_http.post("/_control/reset")).json() == {
        "faults": {},
        "youtube": {"unavailable": False, "frames": "ok", "duration": 600},
        "llm_unavailable": [],
        "llm_rpm": fakes.app.FAKE_LLM_RPM,
    }
    assert (await fakes_http.get("/_control/calls")).json() == []


async def test_the_metadata_can_report_a_longer_video(fakes_http: httpx.AsyncClient) -> None:
    await fakes_http.put("/_control/youtube", json={"duration": 1500})
    assert (await _youtube(fakes_http).video_info(VIDEO_ID)).duration_sec == 1500


async def test_a_bad_switch_is_refused(fakes_http: httpx.AsyncClient) -> None:
    for body in ({"api": "stripe"}, {"api": "llm", "status": 200}, {"api": "llm", "count": 0}):
        assert (await fakes_http.put("/_control/faults", json=body)).status_code == 422
    for switch in ({"frames": "half"}, {"duration": 0}):
        assert (await fakes_http.put("/_control/youtube", json=switch)).status_code == 422
