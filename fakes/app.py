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
from fastapi.responses import StreamingResponse

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


@app.post("/llm/{api_version}/models/{model_action}")
async def generate_content(api_version: str, model_action: str, request: Request) -> dict[str, Any]:
    model, _, action = model_action.partition(":")
    if action != "generateContent":
        raise HTTPException(status_code=404, detail=f"unknown action {action!r}")
    body = await request.json()
    prompt = "\n".join(
        part.get("text", "")
        for content in body.get("contents", [])
        for part in content.get("parts", [])
    )
    schema = body.get("generationConfig", {}).get("responseJsonSchema", {})
    max_frames = schema.get("properties", {}).get("frames", {}).get("maxItems", _DEFAULT_MAX_FRAMES)
    notes = notes_from_transcript(prompt, max_frames)
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


def notes_from_transcript(prompt: str, max_frames: int) -> dict[str, Any]:
    """Deterministic notes: two sections from the transcript lines, evenly spaced frames."""
    lines = [
        (int(h or 0) * 3600 + int(m) * 60 + int(s), text.strip())
        for h, m, s, text in _LINE.findall(prompt)
        if not text.startswith("Candidate frame")  # labels, not spoken lines
    ]
    half = max(1, len(lines) // 2)
    sections = [
        {"heading": f"Part {i + 1}", "bullets": [text for _, text in group[:3]]}
        for i, group in enumerate((lines[:half], lines[half:]))
        if group
    ]
    # Frame-aware prompts (D34) label real frames: pick among those; else among the lines.
    candidates = [int(t) for t in _CANDIDATE.findall(prompt)]
    pool = [(t, "") for t in candidates] or lines
    count = min(max_frames, len(pool))
    picks = [pool[(i * len(pool)) // count] for i in range(count)] if count else []
    return {
        "title": "Fake lecture notes",
        "summary": f"Fake summary of {len(lines)} transcript lines.",
        "sections": sections,
        "frames": [
            {
                "t_sec": t,
                "caption": f"The slide shown at {t // 60}:{t % 60:02d}.",
                "reason": "The speaker refers to the slide here.",
            }
            for t, _ in picks
        ],
    }
