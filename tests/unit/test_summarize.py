"""SUMMARIZE stage: ask, validate, repair, merge, write synthesis.json (ARCHITECTURE §8.2)."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from common.errors import ErrorCode, ExternalApi, PermanentError, RetryableError
from common.models import Synthesis, Transcript, TranscriptSegment, VideoMeta
from common.state import Stage
from fakes.app import app as fakes_app
from tests.unit.fakes import VIDEO_ID, FakeLimiter, make_context
from worker.artifacts import write_json_atomic
from worker.clients.youtube import FakesYouTubeClient
from worker.context import StageContext
from worker.llm.base import LlmRequest
from worker.llm.gemini import GeminiProvider
from worker.stages.summarize import summarize
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

    async def generate_json(self, request: LlmRequest) -> str:
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _context(
    tmp_path: Path, segments: int = 6, progress: list[float] | None = None, **settings: Any
) -> StageContext:
    ctx = make_context(tmp_path, Stage.SUMMARIZE, progress=progress, **settings)
    meta = VideoMeta(video_id=VIDEO_ID, title="Distributed Systems 6", duration_sec=600)
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
    assert outcome.summary == {"chunks": 1, "sections": 1, "frames_requested": 1}
    assert outcome.job_updates == {"frames_requested": 1}
    (request,) = llm.requests
    assert "in English" in request.system
    assert "[0:20] line 1 about Raft" in request.prompt
    frames = request.schema["properties"]["frames"]
    assert (frames["minItems"], frames["maxItems"]) == (3, 8)  # MIN_FRAMES, MAX_FRAMES


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
        model="gemini-3.8-flash",
        fallback_model="gemini-3.5-flash-lite",
        thinking_level="low",
        timeout_s=5,
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
