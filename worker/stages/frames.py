"""EXTRACT_FRAMES stage: download once, ffmpeg seeks, blank check, manifest (ARCHITECTURE §8.3).

Frames only ever come from the source video (C-B). Per pick, FRAME_BURST grabs inside the same
second (t, t+1/n, ...) and the least blurry is kept as frame_{t:05d}.jpg (D33). The burst stops
at a screen change inside that second (D40): the notes' caption describes the picture at t, so
a grab after the change would show something else. A failed or blank frame is a warning, never
a failure (F-20). Idempotent: the download and every kept frame are reused on a re-run.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from common.errors import VnaError, WarningCode
from common.models import FrameResult, JobWarning, Synthesis
from worker.artifacts import write_json
from worker.candidates import Cut, cuts_between
from worker.clients.youtube import VideoDownloader, video_source
from worker.context import StageContext, StageOutcome
from worker.media import FrameGrabError, Grab, grab_frame, sharpest
from worker.stages.summarize import SYNTHESIS_FILE

VIDEO_NAME = "video"
FRAMES_DIR = "frames"
MANIFEST_FILE = "manifest.json"
DOWNLOAD_SHARE = 0.6  # of this stage's progress bar; frame grabs fill the rest
BURST_SPAN_S = 1.0  # every grab stays inside second t (D33)

Grabber = Callable[..., Awaitable[Grab]]
CutFinder = Callable[..., Awaitable[list[Cut]]]


async def run(ctx: StageContext) -> StageOutcome:
    return await extract_frames(ctx, video_source(ctx.settings, ctx.limiter))


async def extract_frames(
    ctx: StageContext,
    source: VideoDownloader,
    grab: Grabber = grab_frame,
    find_cuts: CutFinder = cuts_between,
) -> StageOutcome:
    synthesis = Synthesis.model_validate_json(
        await asyncio.to_thread((ctx.job_dir / SYNTHESIS_FILE).read_text, "utf-8")
    )
    picks = [frame.t_sec for frame in synthesis.frames]
    frames_dir = ctx.job_dir / FRAMES_DIR
    await asyncio.to_thread(frames_dir.mkdir, parents=True, exist_ok=True)

    results: list[FrameResult]
    if not picks:
        results = []  # nothing to capture: no download
    else:
        try:
            video = await source.download_video(
                ctx.job.video_id,
                ctx.job_dir,
                name=VIDEO_NAME,
                max_height=ctx.settings.frames_video_height,
                progress=lambda f: ctx.report_progress(DOWNLOAD_SHARE * f),
            )
        except VnaError as exc:
            if exc.retryable and ctx.message.attempt < ctx.settings.retry_max_attempts:
                raise
            # F-20: the budget is spent (or it can never work): publish text-only, with warnings.
            ctx.log.warning("video_download_gave_up", code=exc.code, attempt=ctx.message.attempt)
            reason = f"video download failed ({exc.code})"
            results = [FrameResult(t_sec=t, error=reason) for t in picks]
        else:
            results = []
            for done, t_sec in enumerate(picks, start=1):
                results.append(await _capture(ctx, video, frames_dir, t_sec, grab, find_cuts))
                await ctx.report_progress(DOWNLOAD_SHARE + (1 - DOWNLOAD_SHARE) * done / len(picks))

    await write_json(frames_dir / MANIFEST_FILE, [r.model_dump(mode="json") for r in results])
    failed = [r for r in results if not r.ok]
    ctx.log.info("frames_saved", requested=len(results), ok=len(results) - len(failed))
    return StageOutcome(
        summary={
            "frames_requested": len(results),
            "frames_ok": len(results) - len(failed),
            "failed": [r.t_sec for r in failed],
        },
        warnings=tuple(
            JobWarning(
                code=WarningCode.FRAME_EXTRACTION_FAILED, timestamp_sec=r.t_sec, message=r.error
            )
            for r in failed
        ),
    )


async def _capture(
    ctx: StageContext,
    video: Path,
    frames_dir: Path,
    t_sec: int,
    grab: Grabber,
    find_cuts: CutFinder,
) -> FrameResult:
    """Burst inside second `t_sec` up to any screen change, keep the least blurry non-blank
    grab, or say why none."""
    name = f"frame_{t_sec:05d}.jpg"
    final = frames_dir / name
    known = await asyncio.to_thread(_kept_before, frames_dir, t_sec)
    if known is not None:
        return known
    settings = ctx.settings
    grabs: list[Grab] = []
    problems: list[str] = []
    offsets = await _burst_offsets(ctx, video, t_sec, find_cuts)
    for k, offset in enumerate(offsets):
        tmp = frames_dir / f"{name}.{k}.tmp"
        try:
            grabbed = await grab(
                video,
                t_sec + offset,
                tmp,
                quality=settings.frame_jpeg_quality,
                timeout_s=settings.ffmpeg_timeout_s,
            )
        except FrameGrabError as exc:
            problems.append(str(exc))
            continue
        blank = grabbed.blank_reason(
            blank_luma=settings.frame_blank_luma, flat_range=settings.frame_flat_range
        )
        if blank:
            problems.append(blank)
        else:
            grabs.append(grabbed)
    try:
        if not grabs:
            # Never substituted (C-B): the frame is simply omitted, with the reason.
            return FrameResult(t_sec=t_sec, error=_main_problem(problems))
        best = sharpest(grabs)
        await asyncio.to_thread(os.replace, best.path, final)
        return FrameResult(
            t_sec=t_sec,
            path=f"{FRAMES_DIR}/{name}",
            offset_s=round(best.at_s - t_sec, 3),
            blur=round(best.blur, 3),
        )
    finally:
        for k in range(settings.frame_burst):
            (frames_dir / f"{name}.{k}.tmp").unlink(missing_ok=True)


async def _burst_offsets(
    ctx: StageContext, video: Path, t_sec: int, find_cuts: CutFinder
) -> list[float]:
    """Offsets into second `t_sec` to grab at: the burst, cut short at the first screen change.
    The grab at t itself (the moment the notes describe) always stays."""
    burst = [k / ctx.settings.frame_burst for k in range(ctx.settings.frame_burst)]
    if len(burst) == 1:
        return burst
    try:
        cuts = await find_cuts(
            video,
            float(t_sec),
            BURST_SPAN_S,
            threshold=ctx.settings.frame_burst_cut_score,
            timeout_s=ctx.settings.ffmpeg_timeout_s,
        )
    except FrameGrabError as exc:
        # Unknown whether the screen changes: keep only the moment the notes describe.
        ctx.log.warning("frame_burst_scan_failed", t_sec=t_sec, error=str(exc)[:200])
        return burst[:1]
    changes = sorted(cut.time_s - t_sec for cut in cuts if cut.time_s > t_sec)
    if not changes:
        return burst
    ctx.log.info("frame_burst_cut", t_sec=t_sec, change_s=round(changes[0], 3))
    return [offset for offset in burst if offset < changes[0]] or burst[:1]


def _kept_before(frames_dir: Path, t_sec: int) -> FrameResult | None:
    """A frame kept by an earlier run of this stage (redelivery), from the old manifest."""
    final = frames_dir / f"frame_{t_sec:05d}.jpg"
    if not final.exists():
        return None
    manifest = frames_dir / MANIFEST_FILE
    if manifest.exists():
        for entry in json.loads(manifest.read_text("utf-8")):
            if entry.get("t_sec") == t_sec and entry.get("path"):
                return FrameResult.model_validate(entry)
    return FrameResult(t_sec=t_sec, path=f"{FRAMES_DIR}/{final.name}")


def _main_problem(problems: list[str]) -> str:
    blank = [p for p in problems if p.startswith("blank frame")]
    return blank[0] if blank else (problems[0] if problems else "no frame")
