"""ffmpeg stats parsing, the blank-frame rule and burst choice (ARCHITECTURE §8.3, D33).

The real-ffmpeg test runs where ffmpeg is installed (the worker image has it; Windows/WSL dev
machines may not) and is otherwise skipped; the fakes-stack integration tests cover it too.
"""

import math
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from worker import media
from worker.media import FrameGrabError, Grab, grab_frame, parse_stats, run_ffmpeg, sharpest

# What `-vf signalstats,blurdetect,metadata=print` printed for one grab in the worker image:
# the filter runs on a second frame too, so every key appears twice.
SAMPLE = """frame:0    pts:10240  pts_time:2
lavfi.signalstats.YMIN=22
lavfi.signalstats.YAVG=125.372
lavfi.signalstats.YMAX=217
lavfi.blur=4.582006
frame:1    pts:10752  pts_time:2.1
lavfi.signalstats.YMIN=23
lavfi.signalstats.YAVG=125.238
lavfi.signalstats.YMAX=220
lavfi.blur=4.608706
"""


def _grab(luma: float, low: float, high: float, blur: float, at_s: float = 0.0) -> Grab:
    return Grab(at_s=at_s, path=Path("x"), luma=luma, luma_min=low, luma_max=high, blur=blur)


def test_stats_come_from_the_first_frame_printed() -> None:
    assert parse_stats(SAMPLE) == {
        "luma_min": 22,
        "luma": 125.372,
        "luma_max": 217,
        "blur": 4.582006,
    }


def test_a_flat_frame_has_no_blur_value() -> None:
    stats = parse_stats("lavfi.signalstats.YAVG=16\nlavfi.blur=-nan\n")
    assert stats["luma"] == 16
    assert math.isnan(stats["blur"])


@pytest.mark.parametrize(
    ("grab", "reason"),
    [
        (_grab(125, 22, 217, 4.6), None),
        (_grab(235, 40, 235, 3.1), None),  # a white slide with dark text is fine
        (_grab(16, 16, 16, math.nan), "blank frame (black)"),  # video black is 16, not 0
        (_grab(23, 16, 60, 9.0), "blank frame (black)"),
        (_grab(230, 222, 235, 2.0), "blank frame (flat)"),  # a fade to white
        (_grab(120, 60, 200, math.nan), "blank frame (flat)"),  # no edges at all
    ],
)
def test_blank_frames_are_recognised(grab: Grab, reason: str | None) -> None:
    assert grab.blank_reason(blank_luma=24, flat_range=24) == reason


def test_the_least_blurry_grab_wins_and_ties_go_to_the_earliest() -> None:
    grabs = [
        _grab(120, 20, 220, 6.0, 42.0),
        _grab(120, 20, 220, 4.5, 42.333),
        _grab(120, 20, 220, 4.5, 42.667),
    ]
    assert sharpest(grabs).at_s == 42.333
    assert sharpest([_grab(120, 20, 220, math.nan, 1.0), _grab(120, 20, 220, 9.0, 1.5)]).at_s == 1.5


class FakeRun:
    """Stands in for subprocess.run; records the call and the thread it ran on."""

    def __init__(self, result: subprocess.CompletedProcess[bytes] | Exception) -> None:
        self.result = result
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.threads: list[str] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((command, kwargs))
        self.threads.append(threading.current_thread().name)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def test_ffmpeg_runs_as_a_blocking_subprocess_in_a_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeRun(subprocess.CompletedProcess(["ffmpeg"], 0, b"lavfi.blur=4.5\n", b"warn\xff"))
    monkeypatch.setattr(media.subprocess, "run", fake)
    run = await run_ffmpeg(["-i", "v.mp4"], timeout_s=12)
    assert (run.returncode, run.stdout, run.stderr) == (0, "lavfi.blur=4.5\n", "warn�")
    [(command, kwargs)] = fake.calls
    assert command[0] == "ffmpeg" and command[-2:] == ["-i", "v.mp4"] and "-nostdin" in command
    assert kwargs["timeout"] == 12 and kwargs["capture_output"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert fake.threads != [threading.main_thread().name]  # never blocks the event loop


async def test_an_ffmpeg_past_its_timeout_is_a_frame_grab_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # subprocess.run has already killed and reaped the child when it raises TimeoutExpired.
    monkeypatch.setattr(media.subprocess, "run", FakeRun(subprocess.TimeoutExpired("ffmpeg", 7)))
    with pytest.raises(FrameGrabError, match="timed out after 7s"):
        await run_ffmpeg(["-i", "v.mp4"], timeout_s=7)


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def test_video(tmp_path: Path) -> Path:
    video = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x180:rate=5:duration=3", "-pix_fmt", "yuv420p", "-y", str(video)],
        check=True,
    )  # fmt: skip
    return video


@needs_ffmpeg
async def test_a_real_grab_reports_brightness_and_blur(tmp_path: Path, test_video: Path) -> None:
    grab = await grab_frame(test_video, 1.0, tmp_path / "f.tmp", quality=3, timeout_s=30)
    assert 16 < grab.luma < 235
    assert grab.blur > 0
    with pytest.raises(FrameGrabError):
        await grab_frame(test_video, 60.0, tmp_path / "late.tmp", quality=3, timeout_s=30)
