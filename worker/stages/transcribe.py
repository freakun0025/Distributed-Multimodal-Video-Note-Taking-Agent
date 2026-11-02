"""TRANSCRIBE stage: metadata, Constraint A, captions or STT (ARCHITECTURE §8.1).

Idempotent: read-only on YouTube; meta.json and transcript.json are overwritten atomically.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from common.errors import ErrorCode, ExternalApi, PermanentError, RetryableError
from common.models import Transcript, VideoMeta
from worker.artifacts import write_json
from worker.clients.youtube import VideoInfo, video_source
from worker.context import StageContext, StageOutcome

META_FILE = "meta.json"
TRANSCRIPT_FILE = "transcript.json"
# Streams that are not a finished video yet: nothing stable to transcribe or capture.
UNSUPPORTED_LIVE_STATUSES = frozenset({"is_live", "is_upcoming", "post_live"})


class VideoSource(Protocol):
    async def video_info(self, video_id: str) -> VideoInfo: ...

    async def transcript(self, video_id: str, languages: Sequence[str]) -> Transcript | None: ...


async def run(ctx: StageContext) -> StageOutcome:
    return await transcribe(ctx, video_source(ctx.settings, ctx.limiter))


async def transcribe(ctx: StageContext, source: VideoSource) -> StageOutcome:
    video_id = ctx.job.video_id
    info = await source.video_info(video_id)
    check_supported(info, ctx.settings.max_video_seconds)
    assert info.duration_sec is not None  # check_supported never passes an unknown length
    await ctx.report_progress(0.4)

    transcript = await source.transcript(video_id, ctx.settings.transcript_langs)
    if transcript is None:
        # STT fallback (STT_ENABLED) arrives in M4 (ROADMAP); until then no captions is final.
        raise PermanentError(ErrorCode.NO_TRANSCRIPT, "this video has no usable captions")
    await ctx.report_progress(0.8)

    meta = VideoMeta(
        video_id=video_id,
        title=info.title,
        duration_sec=info.duration_sec,
        channel=info.channel,
        language=transcript.language or info.language,
    )
    await write_json(ctx.job_dir / META_FILE, meta.model_dump(mode="json"))
    await write_json(ctx.job_dir / TRANSCRIPT_FILE, transcript.model_dump(mode="json"))
    ctx.log.info(
        "transcript_saved",
        source=transcript.source,
        language=transcript.language,
        segments=len(transcript.segments),
        duration_sec=meta.duration_sec,
    )
    return StageOutcome(
        summary={
            "source": transcript.source,
            "language": transcript.language,
            "segments": len(transcript.segments),
            "duration_sec": meta.duration_sec,
        },
        job_updates={"video_title": meta.title, "duration_sec": meta.duration_sec},
    )


def check_supported(info: VideoInfo, max_seconds: int) -> None:
    """Constraint A and live streams: reject deterministically, before any processing."""
    if info.live_status in UNSUPPORTED_LIVE_STATUSES:
        raise PermanentError(
            ErrorCode.VIDEO_UNSUPPORTED, f"live or upcoming stream ({info.live_status})"
        )
    if info.duration_sec is None:
        # Every finished video has a length; its absence is an upstream glitch, so retry (and
        # never process a video whose length Constraint A could not check).
        raise RetryableError(
            ErrorCode.UPSTREAM_ERROR, "YouTube returned no duration", api=ExternalApi.YOUTUBE
        )
    if info.duration_sec > max_seconds:
        raise PermanentError(
            ErrorCode.VIDEO_TOO_LONG,
            f"video is {_clock(info.duration_sec)} long; the limit is {_clock(max_seconds)}",
        )


def _clock(seconds: int) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{minutes}:{rest:02d}"
