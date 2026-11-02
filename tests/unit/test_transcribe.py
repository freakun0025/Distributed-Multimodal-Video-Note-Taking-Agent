"""TRANSCRIBE: Constraint A, live streams, captions → meta.json + transcript.json (§8.1)."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from common.errors import ErrorCode, PermanentError, RetryableError
from common.models import Transcript, TranscriptSegment
from common.state import Stage
from tests.unit.fakes import VIDEO_ID, make_context
from worker.clients.youtube import VideoInfo
from worker.stages.transcribe import transcribe

CAPTIONS = Transcript(
    source="captions",
    language="en",
    segments=[
        TranscriptSegment(start=0.0, duration=4.0, text="Today: consensus."),
        TranscriptSegment(start=4.0, duration=5.5, text="Raft elects a leader."),
    ],
)


class FakeSource:
    def __init__(
        self,
        *,
        duration: int | None = 600,
        live_status: str | None = "not_live",
        transcript: Transcript | None = CAPTIONS,
    ) -> None:
        self.info = VideoInfo(
            video_id=VIDEO_ID,
            title="Distributed Systems, Lecture 6",
            duration_sec=duration,
            live_status=live_status,
            channel="MIT",
            language=None,
        )
        self._transcript = transcript
        self.calls: list[str] = []

    async def video_info(self, video_id: str) -> VideoInfo:
        self.calls.append("video_info")
        return self.info

    async def transcript(self, video_id: str, languages: Sequence[str]) -> Transcript | None:
        self.calls.append(f"transcript:{','.join(languages)}")
        return self._transcript


def _read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


async def test_captions_and_metadata_are_saved_and_the_job_gets_title_and_length(
    tmp_path: Path,
) -> None:
    progress: list[float] = []
    ctx = make_context(tmp_path, Stage.TRANSCRIBE, progress=progress)
    source = FakeSource()

    outcome = await transcribe(ctx, source)

    assert _read(ctx.job_dir / "meta.json") == {
        "video_id": VIDEO_ID,
        "title": "Distributed Systems, Lecture 6",
        "duration_sec": 600,
        "channel": "MIT",
        "language": "en",
    }
    assert _read(ctx.job_dir / "transcript.json") == CAPTIONS.model_dump(mode="json")
    assert outcome.job_updates == {
        "video_title": "Distributed Systems, Lecture 6",
        "duration_sec": 600,
    }
    assert outcome.summary == {
        "source": "captions",
        "language": "en",
        "segments": 2,
        "duration_sec": 600,
    }
    assert progress == [0.4, 0.8]
    assert source.calls == ["video_info", "transcript:en,en-US,en-GB,hi"]


async def test_exactly_twenty_minutes_is_accepted(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, Stage.TRANSCRIBE)
    outcome = await transcribe(ctx, FakeSource(duration=1200))
    assert outcome.job_updates["duration_sec"] == 1200


async def test_a_video_over_twenty_minutes_is_rejected_before_any_processing(
    tmp_path: Path,
) -> None:
    ctx = make_context(tmp_path, Stage.TRANSCRIBE)
    source = FakeSource(duration=1201)
    with pytest.raises(PermanentError) as caught:
        await transcribe(ctx, source)
    assert caught.value.code is ErrorCode.VIDEO_TOO_LONG
    assert caught.value.message == "video is 20:01 long; the limit is 20:00"
    assert source.calls == ["video_info"]  # no captions fetched
    assert not ctx.job_dir.exists()


async def test_the_length_limit_comes_from_settings(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, Stage.TRANSCRIBE, max_video_seconds=300)
    with pytest.raises(PermanentError) as caught:
        await transcribe(ctx, FakeSource(duration=301))
    assert caught.value.code is ErrorCode.VIDEO_TOO_LONG


@pytest.mark.parametrize("status", ["is_live", "is_upcoming", "post_live"])
async def test_live_and_upcoming_streams_are_unsupported(tmp_path: Path, status: str) -> None:
    ctx = make_context(tmp_path, Stage.TRANSCRIBE)
    with pytest.raises(PermanentError) as caught:
        await transcribe(ctx, FakeSource(live_status=status))
    assert caught.value.code is ErrorCode.VIDEO_UNSUPPORTED


async def test_a_finished_live_stream_is_an_ordinary_video(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, Stage.TRANSCRIBE)
    outcome = await transcribe(ctx, FakeSource(live_status="was_live"))
    assert outcome.summary["segments"] == 2


async def test_an_unknown_length_is_retried_never_processed(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, Stage.TRANSCRIBE)
    source = FakeSource(duration=None)
    with pytest.raises(RetryableError) as caught:
        await transcribe(ctx, source)
    assert caught.value.code is ErrorCode.UPSTREAM_ERROR
    assert source.calls == ["video_info"]


async def test_no_captions_fails_the_job_with_no_transcript(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, Stage.TRANSCRIBE)
    with pytest.raises(PermanentError) as caught:
        await transcribe(ctx, FakeSource(transcript=None))
    assert caught.value.code is ErrorCode.NO_TRANSCRIPT
    assert not (ctx.job_dir / "transcript.json").exists()


async def test_running_twice_gives_the_same_artefacts(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, Stage.TRANSCRIBE)
    first = await transcribe(ctx, FakeSource())
    files = {name: _read(ctx.job_dir / name) for name in ("meta.json", "transcript.json")}
    second = await transcribe(ctx, FakeSource())
    assert first == second
    assert {name: _read(ctx.job_dir / name) for name in files} == files
    assert sorted(p.name for p in ctx.job_dir.iterdir()) == ["meta.json", "transcript.json"]
