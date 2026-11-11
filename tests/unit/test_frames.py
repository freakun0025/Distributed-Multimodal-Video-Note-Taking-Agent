"""EXTRACT_FRAMES stage with an injected downloader, grabber and scene scan (§8.3, D33, D40)."""

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pytest

from common.errors import ErrorCode, ExternalApi, PermanentError, RetryableError
from common.models import FramePick, Synthesis
from common.state import Stage
from tests.unit.fakes import VIDEO_ID, make_context
from worker.candidates import Cut
from worker.context import StageContext, StageOutcome
from worker.media import FrameGrabError, Grab
from worker.stages.frames import extract_frames

SHARP, SOFT, BLACK = (120, 20, 220, 4.0), (120, 20, 220, 9.0), (16, 16, 16, math.nan)
MID = (120, 20, 220, 6.0)


class FakeDownloader:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def download_video(
        self, video_id: str, dest_dir: Path, *, name: str, max_height: int, progress: Any
    ) -> Path:
        self.calls.append({"video_id": video_id, "name": name, "max_height": max_height})
        if self.error:
            raise self.error
        await progress(1.0)
        path = dest_dir / f"{name}.mp4"
        path.write_bytes(b"video")
        return path


class FakeGrabber:
    """Stats per (t_sec, burst index); a missing entry means ffmpeg produced nothing."""

    def __init__(self, stats: dict[tuple[int, int], tuple[float, float, float, float]]) -> None:
        self.stats = stats
        self.calls: list[float] = []

    async def __call__(self, video: Path, at_s: float, out: Path, **_: Any) -> Grab:
        self.calls.append(at_s)
        t_sec, k = int(at_s), round((at_s - int(at_s)) * 3)
        if (t_sec, k) not in self.stats:
            raise FrameGrabError("Output file is empty")
        luma, low, high, blur = self.stats[(t_sec, k)]
        await asyncio.to_thread(out.write_bytes, f"jpeg {at_s}".encode())
        return Grab(at_s=at_s, path=out, luma=luma, luma_min=low, luma_max=high, blur=blur)


class FakeCuts:
    """Screen changes (video times) per scanned second; records every scan."""

    def __init__(
        self, changes: dict[int, list[float]] | None = None, error: Exception | None = None
    ) -> None:
        self.changes = changes or {}
        self.error = error
        self.calls: list[dict[str, float]] = []

    async def __call__(
        self, video: Path, start_s: float, span_s: float, *, threshold: float, timeout_s: float
    ) -> list[Cut]:
        self.calls.append({"start_s": start_s, "span_s": span_s, "threshold": threshold})
        if self.error:
            raise self.error
        return [Cut(time_s=t, score=2.6) for t in self.changes.get(int(start_s), [])]


async def _extract(
    ctx: StageContext,
    downloader: FakeDownloader,
    grabber: FakeGrabber,
    cuts: FakeCuts | None = None,
) -> StageOutcome:
    return await extract_frames(ctx, downloader, grabber, cuts or FakeCuts())


def _context(
    tmp_path: Path, picks: list[int], *, attempt: int = 1, progress: list[float] | None = None
) -> StageContext:
    ctx = make_context(tmp_path, Stage.EXTRACT_FRAMES, attempt=attempt, progress=progress)
    synthesis = Synthesis(
        title="t",
        summary="s",
        sections=[],
        questions=[],
        frames=[FramePick(t_sec=t, caption=f"c{t}", reason="r") for t in picks],
    )
    ctx.job_dir.mkdir(parents=True)
    (ctx.job_dir / "synthesis.json").write_text(synthesis.model_dump_json(), "utf-8")
    return ctx


def _manifest(ctx: StageContext) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(
        (ctx.job_dir / "frames/manifest.json").read_text("utf-8")
    )
    return data


async def test_each_pick_keeps_the_sharpest_grab_inside_its_second(tmp_path: Path) -> None:
    progress: list[float] = []
    ctx = _context(tmp_path, [42, 300], progress=progress)
    downloader = FakeDownloader()
    grabber = FakeGrabber(
        {
            (42, 0): SOFT,
            (42, 1): SHARP,
            (42, 2): SOFT,
            (300, 0): SHARP,
            (300, 1): SOFT,
            (300, 2): SOFT,
        }
    )

    outcome = await _extract(ctx, downloader, grabber)

    assert downloader.calls == [{"video_id": VIDEO_ID, "name": "video", "max_height": 720}]
    assert all(int(at) in (42, 300) for at in grabber.calls)  # never outside the picked second
    assert _manifest(ctx) == [
        {
            "t_sec": 42,
            "path": "frames/frame_00042.jpg",
            "error": None,
            "offset_s": 0.333,
            "blur": 4.0,
        },
        {
            "t_sec": 300,
            "path": "frames/frame_00300.jpg",
            "error": None,
            "offset_s": 0.0,
            "blur": 4.0,
        },
    ]
    assert (ctx.job_dir / "frames/frame_00042.jpg").read_bytes() == b"jpeg 42.333333333333336"
    assert sorted(p.name for p in (ctx.job_dir / "frames").iterdir()) == [
        "frame_00042.jpg",
        "frame_00300.jpg",
        "manifest.json",
    ]  # fmt: skip  — no temp files left
    assert outcome.summary == {"frames_requested": 2, "frames_ok": 2, "failed": []}
    assert outcome.warnings == ()
    assert progress[0] == 0.6 and progress[-1] == 1.0


async def test_a_blank_frame_is_omitted_with_a_warning_never_replaced(tmp_path: Path) -> None:
    ctx = _context(tmp_path, [42, 100])
    grabber = FakeGrabber({(42, 0): SHARP, (100, 0): BLACK, (100, 1): BLACK, (100, 2): BLACK})
    outcome = await _extract(ctx, FakeDownloader(), grabber)

    assert _manifest(ctx)[1] == {
        "t_sec": 100,
        "path": None,
        "error": "blank frame (black)",
        "offset_s": None,
        "blur": None,
    }
    assert not (ctx.job_dir / "frames/frame_00100.jpg").exists()
    assert outcome.summary == {"frames_requested": 2, "frames_ok": 1, "failed": [100]}
    (warning,) = outcome.warnings
    assert warning.model_dump() == {
        "code": "FRAME_EXTRACTION_FAILED",
        "timestamp_sec": 100,
        "message": "blank frame (black)",
    }


async def test_a_blank_grab_in_the_burst_is_skipped_for_a_usable_one(tmp_path: Path) -> None:
    ctx = _context(tmp_path, [42])
    outcome = await _extract(ctx, FakeDownloader(), FakeGrabber({(42, 0): BLACK, (42, 2): SOFT}))
    assert _manifest(ctx)[0]["offset_s"] == 0.667
    assert outcome.warnings == ()


async def test_ffmpeg_failing_for_every_grab_is_a_warning(tmp_path: Path) -> None:
    ctx = _context(tmp_path, [42])
    outcome = await _extract(ctx, FakeDownloader(), FakeGrabber({}))
    assert _manifest(ctx)[0]["error"] == "Output file is empty"
    assert outcome.summary["failed"] == [42]


async def test_no_picks_means_no_download(tmp_path: Path) -> None:
    ctx = _context(tmp_path, [])
    downloader = FakeDownloader()
    outcome = await _extract(ctx, downloader, FakeGrabber({}))
    assert downloader.calls == []
    assert _manifest(ctx) == []
    assert outcome.summary == {"frames_requested": 0, "frames_ok": 0, "failed": []}


async def test_a_failed_download_is_retried_while_the_budget_lasts(tmp_path: Path) -> None:
    ctx = _context(tmp_path, [42], attempt=4)  # RETRY_MAX_ATTEMPTS defaults to 5
    error = RetryableError(ErrorCode.UPSTREAM_ERROR, "503", api=ExternalApi.YOUTUBE)
    with pytest.raises(RetryableError):
        await _extract(ctx, FakeDownloader(error), FakeGrabber({}))


@pytest.mark.parametrize(
    ("error", "attempt"),
    [
        (RetryableError(ErrorCode.UPSTREAM_ERROR, "503", api=ExternalApi.YOUTUBE), 5),
        (PermanentError(ErrorCode.VIDEO_UNSUPPORTED, "live", api=ExternalApi.YOUTUBE), 1),
    ],
    ids=["budget spent", "can never work"],
)
async def test_a_download_that_cannot_succeed_publishes_text_only(
    tmp_path: Path, error: Exception, attempt: int
) -> None:
    ctx = _context(tmp_path, [42, 100], attempt=attempt)
    outcome = await _extract(ctx, FakeDownloader(error), FakeGrabber({}))
    assert outcome.summary["frames_ok"] == 0
    assert [w.timestamp_sec for w in outcome.warnings] == [42, 100]
    assert outcome.warnings[0].message == f"video download failed ({error.code})"  # type: ignore[attr-defined]


async def test_a_rerun_keeps_frames_already_saved(tmp_path: Path) -> None:
    ctx = _context(tmp_path, [42, 300])
    first = FakeGrabber({(42, 1): SHARP, (300, 0): SHARP})
    await _extract(ctx, FakeDownloader(), first)
    manifest = _manifest(ctx)

    second = FakeGrabber({})  # would fail every grab
    outcome = await _extract(ctx, FakeDownloader(), second)
    assert second.calls == []
    assert _manifest(ctx) == manifest
    assert outcome.summary["frames_ok"] == 2


async def test_a_screen_change_inside_the_second_ends_the_burst(tmp_path: Path) -> None:
    # The notes describe the screen at 14:02.0; at 14:02.64 the recording switches to another
    # file (a real case, D40). The sharpest grab (+0.667 s) shows that other screen: never taken.
    ctx = _context(tmp_path, [842])
    grabber = FakeGrabber({(842, 0): SOFT, (842, 1): MID, (842, 2): SHARP})
    cuts = FakeCuts({842: [842.64]})

    outcome = await _extract(ctx, FakeDownloader(), grabber, cuts)

    assert cuts.calls == [{"start_s": 842.0, "span_s": 1.0, "threshold": 1.0}]
    assert [round(at, 3) for at in grabber.calls] == [842.0, 842.333]
    assert _manifest(ctx)[0]["offset_s"] == 0.333  # the sharper grab before the change
    assert outcome.warnings == ()


async def test_a_change_right_after_t_keeps_only_the_moment_the_notes_describe(
    tmp_path: Path,
) -> None:
    ctx = _context(tmp_path, [42])
    grabber = FakeGrabber({(42, 0): SOFT, (42, 1): SHARP, (42, 2): SHARP})
    await _extract(ctx, FakeDownloader(), grabber, FakeCuts({42: [42.1]}))
    assert grabber.calls == [42.0]
    assert _manifest(ctx)[0]["offset_s"] == 0.0


async def test_a_change_at_t_itself_is_what_the_notes_saw_so_the_burst_runs(
    tmp_path: Path,
) -> None:
    ctx = _context(tmp_path, [42])
    grabber = FakeGrabber({(42, 0): SOFT, (42, 1): SOFT, (42, 2): SHARP})
    await _extract(ctx, FakeDownloader(), grabber, FakeCuts({42: [42.0]}))
    assert _manifest(ctx)[0]["offset_s"] == 0.667


async def test_a_failed_scene_scan_keeps_only_the_moment_the_notes_describe(
    tmp_path: Path,
) -> None:
    ctx = _context(tmp_path, [42])
    grabber = FakeGrabber({(42, 0): SOFT, (42, 1): SHARP, (42, 2): SHARP})
    cuts = FakeCuts(error=FrameGrabError("scene detection timed out"))
    outcome = await _extract(ctx, FakeDownloader(), grabber, cuts)
    assert grabber.calls == [42.0]
    assert outcome.summary["frames_ok"] == 1 and outcome.warnings == ()
