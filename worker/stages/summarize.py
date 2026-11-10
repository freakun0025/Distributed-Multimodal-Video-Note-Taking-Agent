"""SUMMARIZE stage: LLM synthesis with structured output, chunk + ordered merge (ARCHITECTURE §8.2).

Frame-aware (D34): before the LLM call, real candidate frames from a small download are placed
inside the transcript; the model writes notes from both and picks frames only among them. If
the candidates can't be made (download blocked, ffmpeg failure), the notes come from the
captions alone, as before.

Idempotent: same inputs → synthesis.json overwritten atomically; candidates are cached. A re-run
costs one more LLM call per chunk; the limiter still applies.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

from common.errors import ErrorCode, ExternalApi, PermanentError, VnaError
from common.models import Synthesis, Transcript, VideoMeta
from worker.artifacts import write_json
from worker.candidates import Candidate, CutDetector, Grabber, detect_cuts, prepare_candidates
from worker.clients.youtube import VideoDownloader, video_source
from worker.context import StageContext, StageOutcome
from worker.llm.base import LlmProvider, make_provider
from worker.llm.budget import FrameSlot, NotesFloor, frame_slots, min_frames_for, notes_floor
from worker.llm.chunking import Chunk, chunk_transcript, merge
from worker.llm.prompts import CandidateImage, synthesis_request
from worker.llm.synthesis import (
    Dropped,
    InvalidOutputError,
    finalize,
    parse_part,
    snap_to_candidates,
)
from worker.media import FrameGrabError, grab_frame
from worker.stages.transcribe import META_FILE, TRANSCRIPT_FILE

SYNTHESIS_FILE = "synthesis.json"
CANDIDATES_SHARE = 0.3  # of this stage's progress bar, when candidates are being prepared


async def run(ctx: StageContext) -> StageOutcome:
    return await summarize(
        ctx,
        make_provider(ctx.settings, ctx.limiter, ctx.message.attempt),
        video_source(ctx.settings, ctx.limiter),
    )


async def summarize(
    ctx: StageContext,
    llm: LlmProvider,
    source: VideoDownloader | None = None,
    *,
    detect: CutDetector = detect_cuts,
    grab: Grabber = grab_frame,
) -> StageOutcome:
    meta = VideoMeta.model_validate(_read(ctx.job_dir / META_FILE))
    transcript = Transcript.model_validate(_read(ctx.job_dir / TRANSCRIPT_FILE))
    images = await _candidate_images(ctx, source, meta, detect, grab)
    if source is not None:
        await ctx.report_progress(CANDIDATES_SHARE)  # candidates ready; the LLM call is next
    chunks = chunk_transcript(transcript.segments, ctx.settings.effective_llm_chunk_chars)
    # Spread the frame budget (at least one per MIN_FRAME_EVERY_S, within MIN_FRAMES..MAX_FRAMES)
    # over the parts; the merged result is capped again.
    min_frames = min_frames_for(meta.duration_sec, ctx.settings)
    max_per_chunk = math.ceil(ctx.settings.max_frames / len(chunks))
    min_per_chunk = min(math.ceil(min_frames / len(chunks)), max_per_chunk)
    start = CANDIDATES_SHARE if source is not None else 0.0
    dropped = Dropped()

    parts: list[Synthesis] = []
    for chunk in chunks:
        shown = [c for c in images if chunk.start_sec <= c.t_sec <= chunk.end_sec]
        most = min(max_per_chunk, len(shown)) if shown else max_per_chunk
        least = min(min_per_chunk, most)
        budget = _Budget(
            frames=(least, most),
            notes=notes_floor(chunk.end_sec - chunk.start_sec, ctx.settings),
            # One required frame per stretch of the part, so the pictures cover all of it.
            slots=tuple(
                frame_slots(
                    chunk.start_sec,
                    min(chunk.end_sec, meta.duration_sec - 1),
                    least,
                    [c.t_sec for c in shown],
                )
            ),
        )
        parts.append(await _synthesize(ctx, llm, meta, chunk, len(chunks), budget, shown, dropped))
        await ctx.report_progress(start + (1 - start) * len(parts) / len(chunks))
    synthesis = finalize(
        merge(parts),
        min_gap_s=ctx.settings.frame_min_gap_s,
        max_frames=ctx.settings.max_frames,
        dropped=dropped,
    )

    await write_json(ctx.job_dir / SYNTHESIS_FILE, synthesis.model_dump(mode="json"))
    if dropped.frames:
        ctx.log.info(
            "frames_dropped", frames=[{"t_sec": t, "reason": r} for t, r in dropped.frames]
        )
    ctx.log.info(  # measured on every job: how deep the notes are and how far frames spread
        "synthesis_saved",
        chunks=len(chunks),
        candidates=len(images),
        sections=len(synthesis.sections),
        bullets=sum(len(s.bullets) for s in synthesis.sections),
        words=_words(synthesis),
        frames=len(synthesis.frames),
        largest_gap_s=largest_gap([f.t_sec for f in synthesis.frames], meta.duration_sec),
    )
    return StageOutcome(
        summary={
            "chunks": len(chunks),
            "candidates": len(images),
            "sections": len(synthesis.sections),
            "frames_requested": len(synthesis.frames),
        },
        job_updates={"frames_requested": len(synthesis.frames)},
    )


async def _candidate_images(
    ctx: StageContext,
    source: VideoDownloader | None,
    meta: VideoMeta,
    detect: CutDetector,
    grab: Grabber,
) -> list[CandidateImage]:
    """Candidate frames with their JPEG bytes, or [] (captions-only notes) if unavailable."""
    if source is None or ctx.settings.candidate_max <= 0:
        return []
    try:
        candidates = await prepare_candidates(
            video_id=ctx.job.video_id,
            duration_s=meta.duration_sec,
            job_dir=ctx.job_dir,
            source=source,
            settings=ctx.settings,
            progress=lambda f: ctx.report_progress(CANDIDATES_SHARE * f),
            detect=detect,
            grab=grab,
        )
        return await asyncio.to_thread(_load_images, ctx.job_dir, candidates)
    except (VnaError, FrameGrabError, OSError) as exc:
        code = getattr(exc, "code", None) or type(exc).__name__
        ctx.log.warning("candidates_unavailable", reason=str(code), fallback="captions only")
        return []


def _load_images(job_dir: Path, candidates: list[Candidate]) -> list[CandidateImage]:
    return [
        CandidateImage(t_sec=c.t_sec, data=(job_dir / c.path).read_bytes())
        for c in candidates
        if c.path
    ]


@dataclass(frozen=True)
class _Budget:
    frames: tuple[int, int]  # least and most picks for the part
    notes: NotesFloor  # least sections, and bullets per section
    slots: tuple[FrameSlot, ...]  # where the required picks must come from


async def _synthesize(
    ctx: StageContext,
    llm: LlmProvider,
    meta: VideoMeta,
    chunk: Chunk,
    total: int,
    budget: _Budget,
    shown: list[CandidateImage],
    dropped: Dropped,
) -> Synthesis:
    """One chunk: ask, validate, and re-prompt up to LLM_MAX_REPAIRS times on unusable output."""
    problem: str | None = None
    for attempt in range(ctx.settings.llm_max_repairs + 1):
        request = synthesis_request(
            meta,
            chunk,
            total,
            language=ctx.settings.notes_language,
            min_frames=budget.frames[0],
            max_frames=budget.frames[1],
            floor=budget.notes,
            slots=budget.slots,
            candidates=shown,
            problem=problem,
        )
        text = await llm.generate_json(request)
        try:
            part = parse_part(text, chunk, meta.duration_sec, dropped)
        except InvalidOutputError as exc:
            problem = str(exc)
            ctx.log.warning(
                "llm_output_invalid", chunk=chunk.index, repair=attempt, problem=problem
            )
            continue
        _check_floor(ctx, chunk, part, budget.notes)
        if not shown:
            return part
        return snap_to_candidates(
            part,
            [c.t_sec for c in shown],
            tolerance_s=ctx.settings.candidate_snap_s,
            dropped=dropped,
        )
    raise PermanentError(
        ErrorCode.LLM_INVALID_OUTPUT,
        f"unusable LLM output for part {chunk.index + 1}: {problem}",
        api=ExternalApi.LLM,
    )


def _check_floor(ctx: StageContext, chunk: Chunk, part: Synthesis, floor: NotesFloor) -> None:
    """Gemini enforces the floor while decoding; a provider that doesn't is logged, not
    re-asked (a repair would spend another request of the daily quota on wording)."""
    thin = sum(len(s.bullets) < floor.bullets for s in part.sections)
    if len(part.sections) < floor.sections or thin:
        ctx.log.warning(
            "notes_below_floor",
            chunk=chunk.index,
            sections=len(part.sections),
            thin_sections=thin,
            floor_sections=floor.sections,
            floor_bullets=floor.bullets,
        )


def largest_gap(times: list[int], duration_s: int) -> int:
    """Longest stretch of the video (start and end included) with no frame in it."""
    marks = [0, *sorted(times), max(duration_s, 0)]
    return max(b - a for a, b in itertools.pairwise(marks))


def _words(synthesis: Synthesis) -> int:
    return sum(len(bullet.split()) for s in synthesis.sections for bullet in s.bullets)


def _read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))
