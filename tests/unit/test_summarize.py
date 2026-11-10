"""SUMMARIZE stage: ask, validate, repair, merge, write synthesis.json (ARCHITECTURE §8.2)."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from common.errors import ErrorCode, ExternalApi, PermanentError, RetryableError
from common.models import Synthesis, Transcript, TranscriptSegment, VideoMeta
from common.state import Stage
from fakes.app import app as fakes_app
from fakes.app import notes_from_transcript
from tests.unit.fakes import VIDEO_ID, FakeLimiter, MemoryModelHealth, make_context
from worker.artifacts import write_json_atomic
from worker.candidates import Cut
from worker.clients.fakes import FakesYouTubeClient
from worker.context import StageContext
from worker.llm.base import LlmRequest
from worker.llm.gemini import GeminiProvider
from worker.media import Grab
from worker.stages.summarize import largest_gap, summarize
from worker.stages.transcribe import transcribe

VALID = {
    "title": "Raft",
    "summary": "How Raft reaches consensus.",
    "sections": [{"heading": "Leader election", "bullets": ["Terms", "Votes"]}],
    "frames": [{"t_sec": 30, "caption": "State diagram.", "reason": "Shows the three states."}],
}


class ScriptedLlm:
    """Returns the scripted answers in order; records every request."""

    def __init__(self, *answers: str | Exception) -> None:
        self.answers = list(answers)
        self.requests: list[LlmRequest] = []
        self.answered_by: str | None = "scripted"

    async def generate_json(self, request: LlmRequest) -> str:
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _context(
    tmp_path: Path,
    segments: int = 6,
    progress: list[float] | None = None,
    video_seconds: int = 600,
    **settings: Any,
) -> StageContext:
    ctx = make_context(tmp_path, Stage.SUMMARIZE, progress=progress, **settings)
    meta = VideoMeta(video_id=VIDEO_ID, title="Distributed Systems 6", duration_sec=video_seconds)
    transcript = Transcript(
        source="captions",
        language="en",
        segments=[
            TranscriptSegment(start=i * 20, duration=20, text=f"line {i} about Raft")
            for i in range(segments)
        ],
    )
    write_json_atomic(ctx.job_dir / "meta.json", meta.model_dump(mode="json"))
    write_json_atomic(ctx.job_dir / "transcript.json", transcript.model_dump(mode="json"))
    return ctx


def _synthesis(ctx: StageContext) -> Synthesis:
    return Synthesis.model_validate_json((ctx.job_dir / "synthesis.json").read_text("utf-8"))


async def test_notes_are_written_and_frames_requested_recorded(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    llm = ScriptedLlm(json.dumps(VALID))
    outcome = await summarize(ctx, llm)

    assert _synthesis(ctx) == Synthesis.model_validate(VALID)
    assert outcome.summary == {"chunks": 1, "candidates": 0, "sections": 1, "frames_requested": 1}
    assert outcome.job_updates == {"frames_requested": 1}
    (request,) = llm.requests
    assert "in English" in request.system
    assert "[0:20] line 1 about Raft" in request.prompt
    frames = request.schema["properties"]["frames"]
    assert (frames["minItems"], frames["maxItems"]) == (3, 8)  # MIN_FRAMES, MAX_FRAMES


async def test_an_18_minute_lecture_must_get_deep_notes_and_at_least_6_frames(
    tmp_path: Path,
) -> None:
    # 55 lines x 20 s = 0..1100 s: one part, so its floor is 7 sections x 7 bullets (D37).
    ctx = _context(tmp_path, segments=55, video_seconds=1088)
    llm = ScriptedLlm(json.dumps(VALID))
    await summarize(ctx, llm)

    (request,) = llm.requests
    properties = request.schema["properties"]
    assert (properties["frames"]["minItems"], properties["frames"]["maxItems"]) == (6, 8)
    assert properties["sections"]["minItems"] == 7
    assert properties["sections"]["items"]["properties"]["bullets"]["minItems"] == 7
    assert "at least 7, more if the part covers more topics" in request.system
    assert "between 6 and 8 moments" in request.system
    # One required frame per 3-minute stretch, the last ending at the video's last second.
    stretches = [item["properties"]["t_sec"] for item in properties["frames"]["prefixItems"]]
    assert [(s["minimum"], s["maximum"]) for s in stretches] == [
        (0, 180),
        (181, 361),
        (362, 543),
        (544, 724),
        (725, 905),
        (906, 1087),
    ]


def test_largest_gap_measures_the_longest_stretch_without_a_frame() -> None:
    # The CP7b baseline picks: nothing after 12:20 of an 18:08 video.
    assert largest_gap([45, 166, 438, 589, 740], 1088) == 348
    assert largest_gap([300, 100], 400) == 200
    assert largest_gap([], 600) == 600


def test_the_fake_llm_honours_the_notes_floor_like_gemini() -> None:
    prompt = "\n".join(f"[0:{i:02d}] line {i}" for i in range(10))
    notes = notes_from_transcript(prompt, 4, min_sections=5, min_bullets=4)
    assert len(notes["sections"]) == 5
    assert all(len(section["bullets"]) >= 4 for section in notes["sections"])


async def test_unusable_output_is_repaired_once_with_the_problem_stated(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    llm = ScriptedLlm('{"title": ', json.dumps(VALID))
    await summarize(ctx, llm)
    assert len(llm.requests) == 2
    assert "could not be used (not valid JSON" in llm.requests[1].prompt


async def test_output_still_unusable_after_the_repairs_fails_the_job(tmp_path: Path) -> None:
    ctx = _context(tmp_path, llm_max_repairs=1)
    llm = ScriptedLlm("nope", "[]")
    with pytest.raises(PermanentError) as caught:
        await summarize(ctx, llm)
    assert caught.value.code is ErrorCode.LLM_INVALID_OUTPUT
    assert len(llm.requests) == 2
    assert not (ctx.job_dir / "synthesis.json").exists()


async def test_provider_errors_pass_through_for_retry_routing(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    limited = RetryableError(ErrorCode.RATE_LIMITED, "429", api=ExternalApi.LLM, retry_after_s=30)
    with pytest.raises(RetryableError) as caught:
        await summarize(ctx, ScriptedLlm(limited))
    assert caught.value is limited


async def test_long_transcripts_are_summarised_part_by_part_and_merged_in_order(
    tmp_path: Path,
) -> None:
    progress: list[float] = []
    ctx = _context(tmp_path, segments=30, progress=progress, llm_chunk_chars=300, max_frames=4)

    def part(i: int, t_sec: int) -> str:
        return json.dumps(
            {
                "title": f"Part {i}",
                "summary": f"S{i}.",
                "sections": [{"heading": f"H{i}", "bullets": [f"b{i}"]}],
                "frames": [{"t_sec": t_sec, "caption": "c", "reason": "r"}],
            }
        )

    llm = ScriptedLlm(*(part(i, i * 120) for i in range(30)))
    outcome = await summarize(ctx, llm)

    chunks = outcome.summary["chunks"]
    assert chunks > 1
    assert len(llm.requests) == chunks
    assert [f"Part {i + 1} of {chunks}" in r.prompt for i, r in enumerate(llm.requests)] == [
        True
    ] * chunks
    synthesis = _synthesis(ctx)
    assert [s.heading for s in synthesis.sections] == [f"H{i}" for i in range(chunks)]
    assert synthesis.summary.startswith("S0. S1.")
    # Frame budget spread over the parts (ceil(3 / chunks) .. ceil(4 / chunks)); merged cap 4.
    bounds = {
        (r.schema["properties"]["frames"]["minItems"], r.schema["properties"]["frames"]["maxItems"])
        for r in llm.requests
    }
    assert bounds == {(-(-3 // chunks), -(-4 // chunks))}
    assert len(synthesis.frames) <= 4
    assert progress[-1] == 1.0 and progress == sorted(progress)


async def test_running_twice_with_the_same_answers_writes_the_same_notes(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    await summarize(ctx, ScriptedLlm(json.dumps(VALID)))
    first = (ctx.job_dir / "synthesis.json").read_text("utf-8")
    await summarize(ctx, ScriptedLlm(json.dumps(VALID)))
    assert (ctx.job_dir / "synthesis.json").read_text("utf-8") == first


async def test_fake_apis_path_end_to_end_in_process(tmp_path: Path) -> None:
    """TRANSCRIBE on the fake YouTube, then SUMMARIZE through the real Gemini SDK on the fake."""
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=fakes_app))
    limiter = FakeLimiter()
    ctx = make_context(tmp_path, Stage.TRANSCRIBE, limiter=limiter)
    await transcribe(ctx, FakesYouTubeClient(limiter, "http://fakes", 5, http_client=http))
    llm = GeminiProvider(
        limiter,
        api_key="fake-key",
        chain=ctx.settings.llm_chain,
        health=MemoryModelHealth(),
        base_url="http://fakes/llm",
        http_client=http,
    )
    outcome = await summarize(ctx, llm)

    synthesis = _synthesis(ctx)
    assert synthesis.sections and synthesis.frames
    assert all(0 <= f.t_sec < 600 and f.caption for f in synthesis.frames)
    assert outcome.job_updates == {"frames_requested": len(synthesis.frames)}
    assert [api for api, _ in limiter.calls] == [
        ExternalApi.YOUTUBE,
        ExternalApi.YOUTUBE,
        ExternalApi.LLM,
    ]


class PreviewDownloader:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def download_video(
        self, video_id: str, dest_dir: Path, *, name: str, max_height: int, progress: Any
    ) -> Path:
        if self.error:
            raise self.error
        path = dest_dir / f"{name}.mp4"
        await asyncio.to_thread(path.write_bytes, b"preview")
        return path


async def _no_cuts(video: Path, settings: Any) -> list[Cut]:
    return []


async def _grab_ok(video: Path, at_s: float, out: Path, **_: Any) -> Grab:
    await asyncio.to_thread(out.write_bytes, f"jpeg@{int(at_s)}".encode())
    return Grab(at_s=at_s, path=out, luma=120, luma_min=20, luma_max=220, blur=4.0)


async def test_notes_see_candidate_frames_and_picks_snap_to_them(tmp_path: Path) -> None:
    progress: list[float] = []
    # Transcript: 6 lines over 0-120 s of a 600 s video; 30 s slots -> grid at 15, 45, 75, 105...
    ctx = _context(tmp_path, progress=progress, candidate_max=40, candidate_min_gap_s=30)
    answer = {**VALID, "frames": [{"t_sec": 46, "caption": "The slide.", "reason": "Key."}]}
    llm = ScriptedLlm(json.dumps(answer))

    outcome = await summarize(ctx, llm, PreviewDownloader(), detect=_no_cuts, grab=_grab_ok)

    (request,) = llm.requests
    assert [image.data for image in request.images] == [
        b"jpeg@15",
        b"jpeg@45",
        b"jpeg@75",
        b"jpeg@105",
    ]
    assert "Candidate frame t_sec=45" in request.prompt
    # 3 required frames over 0..120 s: one from each stretch's candidates.
    slots = request.schema["properties"]["frames"]["prefixItems"]
    assert [s["properties"]["t_sec"]["enum"] for s in slots] == [[15], [45, 75], [105]]
    assert [f.t_sec for f in _synthesis(ctx).frames] == [45]  # 46 snapped to the candidate
    assert outcome.summary["candidates"] == 20  # made for the whole video; 4 shown to this part
    assert progress[0] <= 0.3 < progress[-1] == 1.0


async def test_without_candidates_the_notes_come_from_the_captions(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    blocked = RetryableError(ErrorCode.RATE_LIMITED, "bot check", api=ExternalApi.YOUTUBE)
    llm = ScriptedLlm(json.dumps(VALID))
    outcome = await summarize(ctx, llm, PreviewDownloader(blocked), detect=_no_cuts, grab=_grab_ok)
    assert llm.requests[0].images == ()
    assert "only have the transcript" in llm.requests[0].system
    assert outcome.summary["candidates"] == 0
    assert _synthesis(ctx) == Synthesis.model_validate(VALID)
