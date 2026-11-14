"""Video download plumbing: finished-file detection, the fakes client's streamed download, and
the whole-download limit and watchdog stop of both clients (D47)."""

import asyncio
import http.server
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

import fakes.app
from common.errors import ErrorCode, ExternalApi, VnaError
from tests.unit.fakes import VIDEO_ID, FakeLimiter
from worker.clients import youtube
from worker.clients.fakes import FakesYouTubeClient
from worker.clients.youtube import YouTubeClient, finished_download


def test_only_a_complete_download_counts(tmp_path: Path) -> None:
    for partial in (
        "video.mp4.part",
        "video.mp4.ytdl",
        "video.f137.mp4.part",
        "video.mp4.part-Frag3",
    ):
        (tmp_path / partial).write_bytes(b"x")
    assert finished_download(tmp_path, "video") is None
    (tmp_path / "video.webm").write_bytes(b"x")
    assert finished_download(tmp_path, "video") == tmp_path / "video.webm"
    assert finished_download(tmp_path, "preview") is None
    assert finished_download(tmp_path / "missing", "video") is None


@pytest.fixture
def fake_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> bytes:
    content = bytes(range(256)) * 3000  # ~750 kB: several chunks
    path = tmp_path / "fake.mp4"
    path.write_bytes(content)
    monkeypatch.setattr(fakes.app, "VIDEO_PATH", path)
    monkeypatch.setattr(fakes.app, "VIDEO_STREAM_SECONDS", 0.0)
    return content


async def test_the_fake_video_downloads_once_with_progress(
    tmp_path: Path, fake_video: bytes
) -> None:
    limiter = FakeLimiter()
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=fakes.app.app))
    client = FakesYouTubeClient(limiter, "http://fakes", 5, http_client=http)
    progress: list[float] = []

    async def report(fraction: float) -> None:
        progress.append(fraction)

    dest = tmp_path / "job"
    path = await client.download_video(
        VIDEO_ID, dest, name="video", max_height=720, progress=report
    )
    assert path == dest / "video.mp4"
    assert path.read_bytes() == fake_video
    assert (
        progress and progress == sorted(progress) and progress[-1] == pytest.approx(1.0, abs=0.06)
    )
    assert sorted(p.name for p in dest.iterdir()) == ["video.mp4"]  # no partial file left

    again = await client.download_video(
        VIDEO_ID, dest, name="video", max_height=720, progress=report
    )
    assert again == path
    assert limiter.calls == [(ExternalApi.YOUTUBE, 1)]  # the second call reused the file


async def test_the_fake_video_download_stops_at_its_limit(
    tmp_path: Path, fake_video: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fakes.app, "VIDEO_STREAM_SECONDS", 5.0)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=fakes.app.app))
    client = FakesYouTubeClient(
        FakeLimiter(), "http://fakes", 5, http_client=http, download_timeout_s=0.2
    )

    async def report(fraction: float) -> None:
        pass

    dest = tmp_path / "job"
    with pytest.raises(VnaError) as caught:
        await client.download_video(VIDEO_ID, dest, name="video", max_height=720, progress=report)
    assert caught.value.code is ErrorCode.UPSTREAM_TIMEOUT
    assert caught.value.retryable
    assert list(dest.iterdir()) == []  # its partial file is its own and removed


# --- yt-dlp: the whole-download limit and the stage watchdog's stop (D47) ----------------------
# Real yt-dlp against a slow HTTP server on localhost (its generic extractor downloads a plain
# file URL with the same HTTP downloader it uses for YouTube's streams).

SLOW_SIZE = 200_000
SLOW_BLOCK = 4_000


class _SlowVideo(http.server.BaseHTTPRequestHandler):
    """SLOW_SIZE bytes in SLOW_BLOCK blocks, 10 ms apart (~0.5 s in all); honours a Range."""

    def do_GET(self) -> None:
        start = int((self.headers.get("Range") or "bytes=0-").split("=")[1].split("-")[0])
        self.send_response(206 if start else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(SLOW_SIZE - start))
        if start:
            self.send_header("Content-Range", f"bytes {start}-{SLOW_SIZE - 1}/{SLOW_SIZE}")
        self.end_headers()
        try:
            for offset in range(start, SLOW_SIZE, SLOW_BLOCK):
                self.wfile.write(b"v" * min(SLOW_BLOCK, SLOW_SIZE - offset))
                self.wfile.flush()
                time.sleep(0.01)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the client stopped reading

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def slow_video(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """YouTubeClient downloads come from the slow local server instead of YouTube."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SlowVideo)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/video.mp4"
    monkeypatch.setattr(youtube, "watch_url", lambda video_id: url)
    yield
    server.shutdown()
    server.server_close()


async def _ignore(fraction: float) -> None:
    pass


async def _download(client: YouTubeClient, dest: Path) -> Path:
    return await client.download_video(
        VIDEO_ID, dest, name="video", max_height=720, progress=_ignore
    )


async def test_a_download_past_its_limit_stops_and_the_next_attempt_resumes_it(
    tmp_path: Path, slow_video: None
) -> None:
    with pytest.raises(VnaError) as caught:
        await _download(YouTubeClient(FakeLimiter(), 5, download_timeout_s=0.05), tmp_path)
    assert caught.value.code is ErrorCode.UPSTREAM_TIMEOUT
    assert caught.value.retryable
    part = tmp_path / "video.mp4.part"
    assert 0 < part.stat().st_size < SLOW_SIZE  # kept for the next attempt

    path = await _download(YouTubeClient(FakeLimiter(), 5), tmp_path)
    assert path == tmp_path / "video.mp4"
    assert path.stat().st_size == SLOW_SIZE
    assert not part.exists()  # resumed and renamed, nothing left over


async def test_a_cancelled_download_has_stopped_writing_before_the_stage_gives_up(
    tmp_path: Path, slow_video: None
) -> None:
    """The stage watchdog cancels the stage; the retry must not find yt-dlp still writing."""
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.15):
            await _download(YouTubeClient(FakeLimiter(), 5), tmp_path)
    part = tmp_path / "video.mp4.part"
    size = part.stat().st_size
    await asyncio.to_thread(time.sleep, 0.3)  # the server sends ~120 kB in this time
    assert part.stat().st_size == size < SLOW_SIZE

    path = await _download(YouTubeClient(FakeLimiter(), 5), tmp_path)
    assert path.stat().st_size == SLOW_SIZE
