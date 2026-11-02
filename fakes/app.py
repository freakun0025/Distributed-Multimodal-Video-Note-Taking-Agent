"""Fake YouTube/LLM/Notion with switchable 429/5xx/timeouts/frame faults (ARCHITECTURE §14, O-1).

M2 (pulled forward, D29): the happy path `make itest` needs, so integration tests never spend
real YouTube or Gemini quota — YouTube metadata and captions for any video id, and a
Gemini-compatible `generateContent` that writes notes from the transcript it is sent. M3 adds
the failure switches and Notion.
"""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import FastAPI, HTTPException, Request

# Every fake video: a 10-minute lecture with a caption line every 20 s.
DURATION_S = 600
CAPTION_EVERY_S = 20
_LINE = re.compile(r"^\[(?:(\d+):)?(\d+):(\d{2})\] (.*)$", re.MULTILINE)
_DEFAULT_MAX_FRAMES = 3

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
    ]
    half = max(1, len(lines) // 2)
    sections = [
        {"heading": f"Part {i + 1}", "bullets": [text for _, text in group[:3]]}
        for i, group in enumerate((lines[:half], lines[half:]))
        if group
    ]
    count = min(max_frames, len(lines))
    picks = [lines[(i * len(lines)) // count] for i in range(count)] if count else []
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
