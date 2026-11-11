"""Fake YouTube/LLM/Notion with switchable 429/5xx/timeouts/frame faults (ARCHITECTURE §14, O-1).

M2 (pulled forward, D29): the happy path `make itest` needs, so integration tests never spend
real YouTube or Gemini quota — YouTube metadata, captions and a video file for any video id,
and a Gemini-compatible `generateContent` that writes notes from the transcript it is sent. M3
adds the failure switches and Notion.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Every fake video: a 10-minute lecture with a caption line every 20 s.
DURATION_S = 600
CAPTION_EVERY_S = 20
_LINE = re.compile(r"^\[(?:(\d+):)?(\d+):(\d{2})\] (.*)$", re.MULTILINE)
_CANDIDATE = re.compile(r"Candidate frame t_sec=(\d+):")
_DEFAULT_MAX_FRAMES = 3
# Generated when the image is built (docker/fakes.Dockerfile): 600 s, a new colour every 20 s.
VIDEO_PATH = Path(__file__).with_name("video.mp4")
# Served over about this long, like a real download, so EXTRACT_FRAMES has an observable
# window (the kill-a-worker test waits for it).
VIDEO_STREAM_SECONDS = 4.0
_VIDEO_CHUNK = 256 * 1024

app = FastAPI(title="vna fakes", docs_url=None, redoc_url=None)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"fakes": "ok"}


# --- Switches for integration tests (M3 adds 429s, timeouts and frame faults) --------------------

# Models that answer 503 "high demand", as Gemini does under load (model chain test, D36).
_unavailable_models: set[str] = set()
# Every generateContent call since the last switch: model and HTTP status, oldest first.
_llm_calls: list[dict[str, Any]] = []
HIGH_DEMAND = "This model is currently experiencing high demand. Please try again later."


@app.put("/_control/llm")
async def control_llm(body: dict[str, Any]) -> dict[str, Any]:
    """Set the models that answer 503 (an empty list: all answer) and clear the call log."""
    _unavailable_models.clear()
    _unavailable_models.update(body.get("unavailable", []))
    _llm_calls.clear()
    return {"unavailable": sorted(_unavailable_models)}


@app.get("/_control/llm/calls")
async def llm_calls() -> list[dict[str, Any]]:
    return list(_llm_calls)


# --- YouTube ------------------------------------------------------------------------------------


@app.get("/youtube/videos/{video_id}")
async def youtube_video(video_id: str) -> dict[str, Any]:
    """The fields of yt-dlp's info dict that TRANSCRIBE reads."""
    return {
        "id": video_id,
        "title": f"Fake lecture {video_id}",
        "duration": DURATION_S,
        "live_status": "not_live",
        "channel": "Fake channel",
        "language": "en",
    }


@app.get("/youtube/videos/{video_id}/captions")
async def youtube_captions(video_id: str) -> dict[str, Any]:
    segments = [
        {
            "start": float(start),
            "duration": float(CAPTION_EVERY_S),
            "text": f"Point {n} of the fake lecture: on this slide we look at topic {n}.",
        }
        for n, start in enumerate(range(0, DURATION_S, CAPTION_EVERY_S), start=1)
    ]
    return {"language": "en", "segments": segments}


@app.get("/youtube/videos/{video_id}/video")
async def youtube_video_file(video_id: str) -> StreamingResponse:
    """The fake video, streamed in chunks over VIDEO_STREAM_SECONDS."""
    if not VIDEO_PATH.exists():
        raise HTTPException(status_code=404, detail="no fake video in this image")
    size = VIDEO_PATH.stat().st_size
    pause = VIDEO_STREAM_SECONDS / max(1, math.ceil(size / _VIDEO_CHUNK))

    async def body() -> AsyncIterator[bytes]:
        with VIDEO_PATH.open("rb") as handle:
            while chunk := handle.read(_VIDEO_CHUNK):
                yield chunk
                await asyncio.sleep(pause)

    return StreamingResponse(body(), media_type="video/mp4", headers={"content-length": str(size)})


# --- LLM: Gemini generateContent -------------------------------------------------------------


@app.post("/llm/{api_version}/models/{model_action}", response_model=None)
async def generate_content(
    api_version: str, model_action: str, request: Request
) -> dict[str, Any] | JSONResponse:
    model, _, action = model_action.partition(":")
    if action != "generateContent":
        raise HTTPException(status_code=404, detail=f"unknown action {action!r}")
    if model in _unavailable_models:
        _llm_calls.append({"model": model, "status": 503})
        error = {"code": 503, "message": HIGH_DEMAND, "status": "UNAVAILABLE"}
        return JSONResponse(status_code=503, content={"error": error})
    _llm_calls.append({"model": model, "status": 200})
    body = await request.json()
    prompt = "\n".join(
        part.get("text", "")
        for content in body.get("contents", [])
        for part in content.get("parts", [])
    )
    schema = body.get("generationConfig", {}).get("responseJsonSchema", {})
    properties = schema.get("properties", {})
    sections = properties.get("sections", {})
    frames = properties.get("frames", {})
    # Like Gemini, honour the schema: frame budget, notes floor, questions, one frame per slot.
    notes = notes_from_transcript(
        prompt,
        frames.get("maxItems", _DEFAULT_MAX_FRAMES),
        min_sections=sections.get("minItems", 1),
        min_bullets=sections.get("items", {})
        .get("properties", {})
        .get("bullets", {})
        .get("minItems", 1),
        min_questions=properties.get("questions", {}).get("minItems", 1),
        slot_times=[_slot_time(item) for item in frames.get("prefixItems", [])],
    )
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": json.dumps(notes)}]},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": len(prompt) // 4,
            "totalTokenCount": len(prompt) // 4,
        },
        "modelVersion": model,
    }


def _clock(seconds: int) -> str:
    """The transcript's "m:ss", which the real model copies for a section's start."""
    return f"{seconds // 60}:{seconds % 60:02d}"


def _slot_time(item: dict[str, Any]) -> int:
    """A frame slot's middle allowed time: its candidate list, or its range of seconds."""
    t_sec = item.get("properties", {}).get("t_sec", {})
    if choices := t_sec.get("enum"):
        return int(choices[len(choices) // 2])
    return (int(t_sec.get("minimum", 0)) + int(t_sec.get("maximum", 0))) // 2


def notes_from_transcript(
    prompt: str,
    max_frames: int,
    *,
    min_sections: int = 1,
    min_bullets: int = 1,
    min_questions: int = 1,
    slot_times: list[int] | None = None,
) -> dict[str, Any]:
    """Deterministic notes: the transcript lines split into at least two sections (or
    `min_sections`), each starting at its first line's time with at least `min_bullets` bullets;
    `min_questions` self-test questions; one frame per slot when the schema has slots, else
    evenly spaced frames."""
    lines = [
        (int(h or 0) * 3600 + int(m) * 60 + int(s), text.strip())
        for h, m, s, text in _LINE.findall(prompt)
        if not text.startswith("Candidate frame")  # labels, not spoken lines
    ]
    count = max(2, min_sections)
    groups = [
        lines[(i * len(lines)) // count : ((i + 1) * len(lines)) // count] for i in range(count)
    ]
    sections = [
        {
            "heading": f"Part {i + 1}",
            "start": _clock(group[0][0] if group else (lines[-1][0] if lines else 0)),
            "bullets": [text for _, text in group[: max(3, min_bullets)]]
            + [f"Fake point {n + 1}." for n in range(len(group), min_bullets)],
        }
        for i, group in enumerate(groups)
    ]
    # Frame-aware prompts (D34) label real frames: pick among those; else among the lines.
    candidates = [int(t) for t in _CANDIDATE.findall(prompt)]
    pool = [(t, "") for t in candidates] or lines
    count = min(max_frames, len(pool))
    picks = [pool[(i * len(pool)) // count] for i in range(count)] if count else []
    if slot_times:
        picks = [(t, "") for t in slot_times]
    return {
        "title": "Fake lecture notes",
        "summary": f"Fake summary of {len(lines)} transcript lines.",
        "sections": sections,
        "questions": [
            {"question": f"What is fake point {n + 1}?", "answer": f"Fake answer {n + 1}."}
            for n in range(min_questions)
        ],
        "frames": [
            {
                "t_sec": t,
                "caption": f"The slide shown at {t // 60}:{t % 60:02d}.",
                "reason": "The speaker refers to the slide here.",
            }
            for t, _ in picks
        ],
    }
