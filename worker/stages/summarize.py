"""SUMMARIZE stage: LLM synthesis with structured output, chunk + ordered merge (ARCHITECTURE §8.2).

Idempotent: same inputs → synthesis.json overwritten atomically. A re-run costs one more LLM
call per chunk; the limiter still applies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from common.errors import ErrorCode, ExternalApi, PermanentError
from common.models import Synthesis, Transcript, VideoMeta
from worker.artifacts import write_json
from worker.context import StageContext, StageOutcome
from worker.llm.base import LlmProvider, make_provider
from worker.llm.chunking import Chunk, chunk_transcript, merge
from worker.llm.prompts import synthesis_request
from worker.llm.synthesis import Dropped, InvalidOutputError, finalize, parse_part
from worker.stages.transcribe import META_FILE, TRANSCRIPT_FILE

SYNTHESIS_FILE = "synthesis.json"


async def run(ctx: StageContext) -> StageOutcome:
    return await summarize(ctx, make_provider(ctx.settings, ctx.limiter, ctx.message.attempt))


async def summarize(ctx: StageContext, llm: LlmProvider) -> StageOutcome:
    meta = VideoMeta.model_validate(_read(ctx.job_dir / META_FILE))
    transcript = Transcript.model_validate(_read(ctx.job_dir / TRANSCRIPT_FILE))
    chunks = chunk_transcript(transcript.segments, ctx.settings.effective_llm_chunk_chars)
    # Spread the MIN_FRAMES..MAX_FRAMES budget over the parts; the merged result is capped again.
    max_per_chunk = math.ceil(ctx.settings.max_frames / len(chunks))
    min_per_chunk = min(math.ceil(ctx.settings.min_frames / len(chunks)), max_per_chunk)
    dropped = Dropped()

    parts: list[Synthesis] = []
    for chunk in chunks:
        parts.append(
            await _synthesize(
                ctx, llm, meta, chunk, len(chunks), (min_per_chunk, max_per_chunk), dropped
            )
        )
        await ctx.report_progress(len(parts) / len(chunks))
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
    ctx.log.info(
        "synthesis_saved",
        chunks=len(chunks),
        sections=len(synthesis.sections),
        frames=len(synthesis.frames),
    )
    return StageOutcome(
        summary={
            "chunks": len(chunks),
            "sections": len(synthesis.sections),
            "frames_requested": len(synthesis.frames),
        },
        job_updates={"frames_requested": len(synthesis.frames)},
    )


async def _synthesize(
    ctx: StageContext,
    llm: LlmProvider,
    meta: VideoMeta,
    chunk: Chunk,
    total: int,
    frames: tuple[int, int],
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
            min_frames=frames[0],
            max_frames=frames[1],
            problem=problem,
        )
        text = await llm.generate_json(request)
        try:
            return parse_part(text, chunk, meta.duration_sec, dropped)
        except InvalidOutputError as exc:
            problem = str(exc)
            ctx.log.warning(
                "llm_output_invalid", chunk=chunk.index, repair=attempt, problem=problem
            )
    raise PermanentError(
        ErrorCode.LLM_INVALID_OUTPUT,
        f"unusable LLM output for part {chunk.index + 1}: {problem}",
        api=ExternalApi.LLM,
    )


def _read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))
