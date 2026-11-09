"""Candidate frames for frame-aware notes: cuts, slot plan, preparation and cache (D34)."""

import asyncio
import itertools
import json
import math
from pathlib import Path
from typing import Any

from common.config import Settings
from tests.unit.fakes import VIDEO_ID
from worker.candidates import (
    Candidate,
    Cut,
    Grabber,
    parse_cuts,
    plan_candidates,
    prepare_candidates,
)
from worker.media import FrameGrabError, Grab

# `scdet,metadata=select,metadata=print` in the worker image prints only the cut frames.
SCDET_OUTPUT = """frame:160  pts:81920   pts_time:6.66667
lavfi.scd.mafd=10.486
lavfi.scd.score=10.241
lavfi.scd.time=6.66667
frame:742  pts:379904  pts_time:30.9167
lavfi.scd.mafd=72.981
lavfi.scd.score=72.101
lavfi.scd.time=30.9167
"""


def test_cuts_are_parsed_with_their_scores() -> None:
    assert parse_cuts(SCDET_OUTPUT) == [Cut(6.66667, 10.241), Cut(30.9167, 72.101)]
    assert parse_cuts("") == []


def _plan(cuts: list[Cut], duration: int, **options: Any) -> list[Candidate]:
    settings = {"max_count": 36, "min_gap_s": 15, "settle_s": 1.5} | options
    return plan_candidates(cuts, duration, **settings)


def test_without_cuts_a_grid_still_covers_the_whole_video() -> None:
    plan = _plan([], 600)  # 600 // 15 = 40 slots, capped at 36
    assert len(plan) == 36
    assert {c.kind for c in plan} == {"grid"}
    times = [c.t_sec for c in plan]
    assert times[0] < 15 and times[-1] > 580
    assert all(b - a >= 15 for a, b in itertools.pairwise(times))


def test_the_strongest_cut_per_slot_wins_taken_after_it_settles() -> None:
    cuts = [Cut(40.0, 12.0), Cut(44.0, 50.0), Cut(100.0, 30.0)]
    plan = _plan(cuts, 120, max_count=4)  # slots of 30 s
    assert [(c.t_sec, c.kind) for c in plan] == [
        (15, "grid"),
        (45, "cut"),
        (75, "grid"),
        (101, "cut"),
    ]
    assert plan[1].score == 50.0  # 44 + 1.5 = 45.5 -> second 45


def test_settling_stops_before_the_next_cut() -> None:
    plan = _plan([Cut(30.2, 40.0), Cut(31.0, 5.0)], 60, max_count=2)
    assert plan[1].t_sec == 30  # 30.2 + 1.5 would pass the next cut at 31.0: capped at 30.5


def test_candidates_keep_the_minimum_gap_and_stay_inside_the_video() -> None:
    cuts = [Cut(29.0, 90.0), Cut(31.0, 90.0), Cut(59.5, 90.0)]
    plan = _plan(cuts, 60, max_count=2, min_gap_s=15)  # slots [0,30) and [30,60)
    times = [c.t_sec for c in plan]
    assert times[1] - times[0] >= 15
    assert all(t <= 59 for t in times)
    assert _plan([], 10) == [Candidate(t_sec=5, kind="grid")]  # a short video: one slot
    assert _plan([Cut(1.0, 9.0)], 60, max_count=0) == []


class FakeDownloader:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def download_video(
        self, video_id: str, dest_dir: Path, *, name: str, max_height: int, progress: Any
    ) -> Path:
        self.calls.append({"video_id": video_id, "name": name, "max_height": max_height})
        path = dest_dir / f"{name}.mp4"
        await asyncio.to_thread(path.write_bytes, b"preview")
        return path


def _grabber(
    blank_at: frozenset[int] = frozenset(), broken_at: frozenset[int] = frozenset()
) -> Grabber:
    async def grab(video: Path, at_s: float, out: Path, **_: Any) -> Grab:
        if int(at_s) in broken_at:
            raise FrameGrabError("no frame")
        await asyncio.to_thread(out.write_bytes, b"\xff\xd8")
        if int(at_s) in blank_at:
            return Grab(at_s=at_s, path=out, luma=16, luma_min=16, luma_max=16, blur=math.nan)
        return Grab(at_s=at_s, path=out, luma=120, luma_min=20, luma_max=220, blur=4.0)

    return grab


async def test_candidates_are_prepared_once_and_blank_ones_left_out(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, candidate_max=4)  # type: ignore[call-arg]
    downloader = FakeDownloader()

    async def detect(video: Path, _: Settings) -> list[Cut]:
        assert video.name == "preview.mp4"
        return [Cut(44.0, 50.0)]

    async def progress(_: float) -> None:
        pass

    kwargs: dict[str, Any] = {
        "video_id": VIDEO_ID, "duration_s": 120, "job_dir": tmp_path, "source": downloader,
        "settings": settings, "progress": progress, "detect": detect,
    }  # fmt: skip
    first = await prepare_candidates(
        **kwargs, grab=_grabber(blank_at=frozenset({75}), broken_at=frozenset({15}))
    )
    assert [(c.t_sec, c.kind, c.path) for c in first] == [
        (45, "cut", "candidates/cand_00045.jpg"),
        (105, "grid", "candidates/cand_00105.jpg"),
    ]
    assert downloader.calls == [{"video_id": VIDEO_ID, "name": "preview", "max_height": 480}]
    assert sorted(p.name for p in (tmp_path / "candidates").iterdir()) == [
        "cand_00045.jpg",
        "cand_00105.jpg",
        "manifest.json",
    ]  # fmt: skip — the blank grab was deleted
    assert json.loads((tmp_path / "candidates/manifest.json").read_text("utf-8"))[0]["t_sec"] == 45

    again = await prepare_candidates(**kwargs, grab=_grabber(broken_at=frozenset(range(200))))
    assert again == first
    assert len(downloader.calls) == 1  # cached: no second download


async def test_candidate_max_zero_means_captions_only(tmp_path: Path) -> None:
    downloader = FakeDownloader()

    async def progress(_: float) -> None:
        pass

    settings = Settings(_env_file=None, candidate_max=0)  # type: ignore[call-arg]
    result = await prepare_candidates(
        video_id=VIDEO_ID, duration_s=120, job_dir=tmp_path, source=downloader,
        settings=settings, progress=progress,
    )  # fmt: skip
    assert result == []
    assert downloader.calls == []
