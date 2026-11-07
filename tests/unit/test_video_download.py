"""Video download plumbing: finished-file detection and the fakes client's streamed download."""

from pathlib import Path

import httpx
import pytest

import fakes.app
from common.errors import ExternalApi
from tests.unit.fakes import VIDEO_ID, FakeLimiter
from worker.clients.fakes import FakesYouTubeClient
from worker.clients.youtube import finished_download


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
