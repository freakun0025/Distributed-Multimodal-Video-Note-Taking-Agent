"""Candidate frames for frame-aware notes (ARCHITECTURE §8.2, D34).

SUMMARIZE downloads a small copy of the video once, finds scene cuts with ffmpeg's `scdet`,
and keeps at most CANDIDATE_MAX real frames spread over the whole video: per time slot the
strongest cut (grabbed CANDIDATE_SETTLE_S later, once a fade or slide build has settled), else
the slot's middle, never closer than CANDIDATE_MIN_GAP_S, never blank. The LLM sees these frames
inside the transcript and picks its frames only among them. Cached under `candidates/`.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from common.config import Settings
from common.logging import get_logger
from worker.artifacts import write_json_atomic
from worker.clients.youtube import ProgressFn, VideoDownloader
from worker.media import FrameGrabError, Grab, grab_frame, run_ffmpeg

CANDIDATES_DIR = "candidates"
MANIFEST_FILE = "manifest.json"
PREVIEW_NAME = "preview"
GRAB_CONCURRENCY = 4  # ffmpeg processes at once per worker
log = get_logger()

# Only frames scdet marks as cuts are printed: their score, then their time.
_SCDET_FILTER = (
    "scdet=threshold={threshold},metadata=mode=select:key=lavfi.scd.time,metadata=mode=print:file=-"
)

Grabber = Callable[..., Awaitable[Grab]]
CutDetector = Callable[[Path, Settings], Awaitable[list["Cut"]]]


@dataclass(frozen=True)
class Cut:
    time_s: float
    score: float


@dataclass(frozen=True)
class Candidate:
    t_sec: int
    kind: Literal["cut", "grid"]
    score: float | None = None
    path: str | None = None  # relative to the job directory, once grabbed


def parse_cuts(output: str) -> list[Cut]:
    cuts: list[Cut] = []
    score: float | None = None
    for line in output.splitlines():
        key, _, value = line.strip().partition("=")
        if key == "lavfi.scd.score":
            score = float(value)
        elif key == "lavfi.scd.time" and score is not None:
            cuts.append(Cut(time_s=float(value), score=score))
            score = None
    return cuts


async def detect_cuts(video: Path, settings: Settings) -> list[Cut]:
    """Scene cuts over the whole video (decodes every frame: ~8 s for 11 min at 480p)."""
    return await _scdet(
        ["-i", str(video)], settings.scene_threshold, settings.scene_detect_timeout_s
    )


async def cuts_between(
    video: Path, start_s: float, span_s: float, *, threshold: float, timeout_s: float
) -> list[Cut]:
    """Scene changes scoring at least `threshold` in [start_s, start_s + span_s), at video
    times. EXTRACT_FRAMES uses it to keep a frame burst on the screen the notes describe."""
    window = ["-ss", f"{start_s:.3f}", "-t", f"{span_s:.3f}", "-i", str(video)]
    cuts = await _scdet(window, threshold, timeout_s)
    # Input seeking restarts timestamps at 0, so scdet's times are relative to the window.
    return [Cut(time_s=start_s + cut.time_s, score=cut.score) for cut in cuts]


async def _scdet(source: list[str], threshold: float, timeout_s: float) -> list[Cut]:
    run = await run_ffmpeg(
        [*source, "-an", "-vf", _SCDET_FILTER.format(threshold=threshold), "-f", "null", "-"],
        timeout_s,
    )
    if run.returncode != 0:
        raise FrameGrabError(run.stderr.strip()[-200:] or "scdet failed")
    return parse_cuts(run.stdout)


def plan_candidates(
    cuts: list[Cut], duration_s: int, *, max_count: int, min_gap_s: int, settle_s: float
) -> list[Candidate]:
    """Up to `max_count` whole-second times spread over the video, cuts preferred."""
    last_second = duration_s - 1
    if max_count <= 0 or last_second < 0:
        return []
    slots = max(1, min(max_count, duration_s // min_gap_s))
    width = duration_s / slots
    ordered = sorted(cuts, key=lambda c: c.time_s)
    chosen: list[Candidate] = []
    for i in range(slots):
        start, end = i * width, (i + 1) * width
        options = [
            Candidate(t_sec=int(_settled(cut, ordered, settle_s)), kind="cut", score=cut.score)
            for cut in sorted(
                (c for c in ordered if start <= c.time_s < end), key=lambda c: -c.score
            )
        ]
        options.append(Candidate(t_sec=int((start + end) / 2), kind="grid"))
        for option in options:
            if option.t_sec > last_second:
                continue
            if chosen and option.t_sec - chosen[-1].t_sec < min_gap_s:
                continue
            chosen.append(option)
            break
    return chosen


def _settled(cut: Cut, ordered: list[Cut], settle_s: float) -> float:
    """`settle_s` after the cut, but before the next cut begins."""
    later = [c.time_s for c in ordered if c.time_s > cut.time_s]
    when = cut.time_s + settle_s
    if later:
        when = min(when, max(cut.time_s, later[0] - 0.5))
    return when


async def prepare_candidates(
    *,
    video_id: str,
    duration_s: int,
    job_dir: Path,
    source: VideoDownloader,
    settings: Settings,
    progress: ProgressFn,
    detect: CutDetector = detect_cuts,
    grab: Grabber = grab_frame,
) -> list[Candidate]:
    """The cached candidates, or download + detect + grab them now. Raises on failure."""
    folder = job_dir / CANDIDATES_DIR
    manifest = folder / MANIFEST_FILE
    cached = await asyncio.to_thread(_load, manifest)
    if cached is not None:
        return cached
    if settings.candidate_max <= 0:
        return []
    started = time.monotonic()
    preview = await source.download_video(
        video_id,
        job_dir,
        name=PREVIEW_NAME,
        max_height=settings.candidates_video_height,
        progress=progress,
    )
    downloaded = time.monotonic()
    cuts = await detect(preview, settings)
    detected = time.monotonic()
    planned = plan_candidates(
        cuts,
        duration_s,
        max_count=settings.candidate_max,
        min_gap_s=settings.candidate_min_gap_s,
        settle_s=settings.candidate_settle_s,
    )
    await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
    gate = asyncio.Semaphore(GRAB_CONCURRENCY)

    async def thumbnail(candidate: Candidate) -> Candidate | None:
        name = f"cand_{candidate.t_sec:05d}.jpg"
        async with gate:
            try:
                grabbed = await grab(
                    preview,
                    float(candidate.t_sec),
                    folder / name,
                    quality=settings.candidate_jpeg_quality,
                    timeout_s=settings.ffmpeg_timeout_s,
                )
            except FrameGrabError:
                return None
        if grabbed.blank_reason(
            blank_luma=settings.frame_blank_luma, flat_range=settings.frame_flat_range
        ):
            await asyncio.to_thread((folder / name).unlink, missing_ok=True)
            return None
        return Candidate(
            t_sec=candidate.t_sec,
            kind=candidate.kind,
            score=candidate.score,
            path=f"{CANDIDATES_DIR}/{name}",
        )

    grabbed = [c for c in await asyncio.gather(*map(thumbnail, planned)) if c is not None]
    await asyncio.to_thread(_save, manifest, grabbed)
    log.info(  # measured on every job, not estimated (CP7b)
        "candidates_ready",
        cuts=len(cuts),
        planned=len(planned),
        kept=len(grabbed),
        from_cuts=sum(c.kind == "cut" for c in grabbed),
        download_s=round(downloaded - started, 1),
        detect_s=round(detected - downloaded, 1),
        grab_s=round(time.monotonic() - detected, 1),
    )
    return grabbed


def _load(manifest: Path) -> list[Candidate] | None:
    if not manifest.exists():
        return None
    return [Candidate(**entry) for entry in json.loads(manifest.read_text("utf-8"))]


def _save(manifest: Path, candidates: list[Candidate]) -> None:
    write_json_atomic(manifest, [asdict(c) for c in candidates])
