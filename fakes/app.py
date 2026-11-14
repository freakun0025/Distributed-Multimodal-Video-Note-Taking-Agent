"""Fake YouTube/LLM/Notion with switchable failures (ARCHITECTURE §14, O-1).

The happy path (D29) lets integration tests and demos run without spending real YouTube, Gemini
or Notion quota: metadata, captions and a video file for any video id, a Gemini-compatible
`generateContent` that writes notes from the transcript it is sent, and a Notion workspace.

The `/_control/*` switches reproduce every Spec §6 failure on demand (CP9): the next N calls to
one API answer an error (429 by default, with the real API's error body, optional Retry-After),
every video is private, or the frames download is cut short or black. The worker's real clients
classify these answers exactly as they classify the real APIs' (`make fake-429` etc.).

Like Gemini's free tier, the fake LLM answers 429 to a model's call past FAKE_LLM_RPM in the
last 60 s (CP11), so "a burst drains with zero LLM 429s" is measured against a real limit.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Every fake video: a 10-minute lecture with a caption line every 20 s.
DURATION_S = 600
CAPTION_EVERY_S = 20
_LINE = re.compile(r"^\[(?:(\d+):)?(\d+):(\d{2})\] (.*)$", re.MULTILINE)
_CANDIDATE = re.compile(r"Candidate frame t_sec=(\d+):")
_DEFAULT_MAX_FRAMES = 3
# Generated when the image is built (docker/fakes.Dockerfile): 600 s, a new colour every 20 s.
VIDEO_PATH = Path(__file__).with_name("video.mp4")
# Frame faults (CP9), also built with the image: the same video cut at CUT_AT_S (picks after it
# get no frame), and 600 s of black (every grab is blank).
CUT_AT_S = 300  # `-t 300` in docker/fakes.Dockerfile
FRAME_FAULT_VIDEOS = {
    "partial": Path(__file__).with_name("video_half.mp4"),
    "none": Path(__file__).with_name("video_black.mp4"),
}
# Frame faults hit only downloads taller than this: EXTRACT_FRAMES' (FRAMES_VIDEO_HEIGHT, 720),
# not SUMMARIZE's candidate preview (CANDIDATES_VIDEO_HEIGHT, 480), so the notes still pick
# frames over the whole video and the frame stage then fails some or all of them.
PREVIEW_MAX_HEIGHT = 480
# Served over about this long, like a real download, so EXTRACT_FRAMES has an observable
# window (the kill-a-worker test waits for it).
VIDEO_STREAM_SECONDS = 4.0
# The captions take about this long, like a real fetch, so TRANSCRIBE has an observable window
# too (the kill-a-worker test kills a worker in every stage, CP10).
CAPTIONS_SECONDS = 2.0
_VIDEO_CHUNK = 256 * 1024
# yt-dlp's message for a private video (the real client classifies it VIDEO_UNAVAILABLE, D12).
PRIVATE_VIDEO = "Private video. Sign in if you've been granted access to this video"

app = FastAPI(title="vna fakes", docs_url=None, redoc_url=None)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"fakes": "ok"}


# --- Failure switches (Spec §6 on demand; make fake-429 / fake-frames / fake-unavailable) ------

Api = Literal["youtube", "llm", "notion"]
FrameMode = Literal["ok", "partial", "none"]
_CALL_LOG_MAX = 2000

# Models that answer 503 "high demand", as Gemini does under load (model chain test, D36).
_unavailable_models: set[str] = set()
# Every generateContent call since the last switch: model and HTTP status, oldest first.
_llm_calls: list[dict[str, Any]] = []
HIGH_DEMAND = "This model is currently experiencing high demand. Please try again later."
# Calls a model answers per 60 s before it answers 429, like Gemini's per-minute quota:
# gemini-3.8-flash and gemini-3.6-flash allow 5 on the free tier (ARCHITECTURE §9). 0 = no limit.
FAKE_LLM_RPM = 5
RPM_WINDOW_S = 60.0
_llm_rpm = FAKE_LLM_RPM
_llm_served: dict[str, deque[float]] = {}  # model -> monotonic times of answered calls
_monotonic = time.monotonic  # replaced in unit tests to move the minute on


class FaultSwitch(BaseModel):
    """The next `count` calls to `api` answer `status` instead of working."""

    api: Api
    status: int = Field(default=429, ge=400, le=599)
    count: int = Field(default=1, ge=1, le=1000)
    retry_after: int | None = Field(default=None, ge=0, le=3600)  # seconds


class YouTubeSwitch(BaseModel):
    """Omitted fields keep their current value."""

    unavailable: bool | None = None  # every video private: TRANSCRIBE retries, then the DLQ
    frames: FrameMode | None = None  # the frames download: intact, cut at 300 s, or black
    duration: int | None = Field(default=None, gt=0)  # seconds in the metadata (Constraint A)


@dataclass
class _Fault:
    status: int
    remaining: int
    retry_after: int | None


_faults: dict[str, _Fault] = {}
_YOUTUBE_DEFAULTS: dict[str, Any] = {"unavailable": False, "frames": "ok", "duration": DURATION_S}
_youtube: dict[str, Any] = dict(_YOUTUBE_DEFAULTS)
# Every YouTube/LLM/Notion call, oldest first: api, method, path and the status answered.
_calls: deque[dict[str, Any]] = deque(maxlen=_CALL_LOG_MAX)


def _api_of(path: str) -> Api | None:
    """Which fake API a request is for; the OAuth exchange (gateway, not the pipeline) is none."""
    if path.startswith("/youtube/"):
        return "youtube"
    if path.startswith("/llm/"):
        return "llm"
    if path.startswith("/notion/") and not path.startswith("/notion/v1/oauth/"):
        return "notion"
    return None


@app.middleware("http")
async def inject_faults(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    api = _api_of(request.url.path)
    if api is None:
        return await call_next(request)
    fault = _faults.get(api)
    if fault is not None and fault.remaining > 0:
        fault.remaining -= 1
        response: Response = _fault_response(api, fault.status, fault.retry_after)
    else:
        response = await call_next(request)
    _calls.append(
        {
            "api": api,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
        }
    )
    return response


def _fault_response(api: Api, status: int, retry_after: int | None) -> JSONResponse:
    """The error each real API sends, so the worker's clients take their real code paths."""
    try:
        phrase = HTTPStatus(status).phrase
    except ValueError:
        phrase = "Error"
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    if api == "llm":
        return JSONResponse(status_code=status, content=_gemini_error(status, phrase, retry_after))
    if api == "notion":
        code, message = _NOTION_ERRORS.get(status, ("internal_server_error", phrase))
        body = {"object": "error", "status": status, "code": code, "message": message}
        return JSONResponse(status_code=status, content=body, headers=headers)
    return JSONResponse(
        status_code=status, content={"error": {"code": status, "message": phrase}}, headers=headers
    )


_NOTION_ERRORS = {
    429: ("rate_limited", "You have been rate limited. Please try again in a few minutes."),
    502: ("bad_gateway", "Notion encountered an issue while attempting to complete this request."),
    503: ("service_unavailable", "Notion is unavailable, please try again later."),
    504: ("gateway_timeout", "Notion timed out while attempting to complete this request."),
}
_GOOGLE_STATUS = {
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    503: "UNAVAILABLE",
    504: "DEADLINE_EXCEEDED",
}


def _gemini_error(status: int, phrase: str, retry_after: int | None) -> dict[str, Any]:
    """Gemini's error body. A 429 is the per-minute quota (retried after a delay); a per-day
    quota 429 would bench the model instead (D36) and is not what this switch is for."""
    error: dict[str, Any] = {
        "code": status,
        "message": HIGH_DEMAND if status == 503 else phrase,
        "status": _GOOGLE_STATUS.get(status, "UNKNOWN"),
    }
    if status == 429:
        error["message"] = "You exceeded your current quota. Please check your plan and billing."
        quota = {
            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
            "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
        }
        error["details"] = [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [quota]}
        ]
        if retry_after is not None:
            retry = {"@type": "type.googleapis.com/google.rpc.RetryInfo"}
            error["details"].append({**retry, "retryDelay": f"{retry_after}s"})
    return {"error": error}


def _switches() -> dict[str, Any]:
    return {
        "faults": {
            api: {"status": f.status, "remaining": f.remaining, "retry_after": f.retry_after}
            for api, f in _faults.items()
            if f.remaining > 0
        },
        "youtube": dict(_youtube),
        "llm_unavailable": sorted(_unavailable_models),
        "llm_rpm": _llm_rpm,
    }


@app.put("/_control/faults")
async def control_faults(switch: FaultSwitch) -> dict[str, Any]:
    """Replace `api`'s pending error answers with `count` answers of `status`."""
    _faults[switch.api] = _Fault(switch.status, switch.count, switch.retry_after)
    return _switches()


@app.put("/_control/youtube")
async def control_youtube(switch: YouTubeSwitch) -> dict[str, Any]:
    _youtube.update(switch.model_dump(exclude_none=True))
    return _switches()


@app.get("/_control/switches")
async def switches() -> dict[str, Any]:
    return _switches()


@app.post("/_control/reset")
async def reset() -> dict[str, Any]:
    """Every API works again and the call log is empty (the Notion workspace is kept); the LLM
    is back to FAKE_LLM_RPM with an empty minute."""
    global _llm_rpm
    _faults.clear()
    _youtube.update(_YOUTUBE_DEFAULTS)
    _unavailable_models.clear()
    _llm_calls.clear()
    _calls.clear()
    _llm_rpm = FAKE_LLM_RPM
    _llm_served.clear()
    return _switches()


@app.get("/_control/calls")
async def calls(api: Api | None = None) -> list[dict[str, Any]]:
    return [call for call in _calls if api is None or call["api"] == api]


@app.put("/_control/llm")
async def control_llm(body: dict[str, Any]) -> dict[str, Any]:
    """Set the models that answer 503 (an empty list: all answer) and clear the call log;
    `rpm` (optional) sets the per-model calls per minute (0: no limit)."""
    global _llm_rpm
    _unavailable_models.clear()
    _unavailable_models.update(body.get("unavailable", []))
    _llm_calls.clear()
    if "rpm" in body:
        rpm = body["rpm"]
        if not isinstance(rpm, int) or isinstance(rpm, bool) or rpm < 0:
            raise HTTPException(status_code=422, detail="rpm must be an integer >= 0")
        _llm_rpm = rpm
        _llm_served.clear()
    return {"unavailable": sorted(_unavailable_models), "rpm": _llm_rpm}


def _over_rpm(model: str) -> int | None:
    """Seconds until `model` may be called again when its last minute is full, else None (and
    the call counts toward the minute)."""
    if _llm_rpm <= 0:
        return None
    now = _monotonic()
    served = _llm_served.setdefault(model, deque())
    while served and now - served[0] >= RPM_WINDOW_S:
        served.popleft()
    if len(served) >= _llm_rpm:
        return max(1, math.ceil(RPM_WINDOW_S - (now - served[0])))
    served.append(now)
    return None


@app.get("/_control/llm/calls")
async def llm_calls() -> list[dict[str, Any]]:
    return list(_llm_calls)


# --- YouTube ------------------------------------------------------------------------------------


def _private() -> JSONResponse:
    return JSONResponse(status_code=403, content={"error": {"code": 403, "message": PRIVATE_VIDEO}})


@app.get("/youtube/videos/{video_id}", response_model=None)
async def youtube_video(video_id: str) -> dict[str, Any] | JSONResponse:
    """The fields of yt-dlp's info dict that TRANSCRIBE reads."""
    if _youtube["unavailable"]:
        return _private()
    return {
        "id": video_id,
        "title": f"Fake lecture {video_id}",
        "duration": _youtube["duration"],
        "live_status": "not_live",
        "channel": "Fake channel",
        "language": "en",
    }


@app.get("/youtube/videos/{video_id}/captions", response_model=None)
async def youtube_captions(video_id: str) -> dict[str, Any] | JSONResponse:
    if _youtube["unavailable"]:
        return _private()
    await asyncio.sleep(CAPTIONS_SECONDS)
    segments = [
        {
            "start": float(start),
            "duration": float(CAPTION_EVERY_S),
            "text": f"Point {n} of the fake lecture: on this slide we look at topic {n}.",
        }
        for n, start in enumerate(range(0, DURATION_S, CAPTION_EVERY_S), start=1)
    ]
    return {"language": "en", "segments": segments}


@app.get("/youtube/videos/{video_id}/video", response_model=None)
async def youtube_video_file(
    video_id: str, height: int | None = None
) -> StreamingResponse | JSONResponse:
    """The fake video at most `height` pixels tall, streamed in chunks over VIDEO_STREAM_SECONDS.
    With a frame fault on, a download taller than the preview gets the broken video."""
    if _youtube["unavailable"]:
        return _private()
    path = VIDEO_PATH
    if _youtube["frames"] != "ok" and (height is None or height > PREVIEW_MAX_HEIGHT):
        path = FRAME_FAULT_VIDEOS[_youtube["frames"]]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no {path.name} in this image")
    size = path.stat().st_size
    pause = VIDEO_STREAM_SECONDS / max(1, math.ceil(size / _VIDEO_CHUNK))

    async def body() -> AsyncIterator[bytes]:
        with path.open("rb") as handle:
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
    if (wait_s := _over_rpm(model)) is not None:
        _llm_calls.append({"model": model, "status": 429})
        return JSONResponse(status_code=429, content=_gemini_error(429, "", wait_s))
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


# --- Notion: the calls PUBLISH and the gateway's OAuth exchange make (ARCHITECTURE §8.4, §11) ---

# One in-memory workspace: every block (pages included) by id, and each parent's children in order.
_notion_blocks: dict[str, dict[str, Any]] = {}
_notion_children: dict[str, list[str]] = {}
_notion_uploads: dict[str, str] = {}  # file upload id -> "pending" | "uploaded"
_NOTION_MAX_CHILDREN = 100
BAD_OAUTH_CODE = "bad-code"  # the code Notion rejects (invalid_grant)


def _notion_error(status: int, code: str, message: str) -> JSONResponse:
    body = {"object": "error", "status": status, "code": code, "message": message}
    return JSONResponse(status_code=status, content=body)


def _signed_in(request: Request) -> bool:
    auth = request.headers.get("authorization", "")
    return auth.startswith("Bearer ") and len(auth) > len("Bearer ")


def _new_id() -> str:
    return str(uuid.uuid4())


@app.put("/_control/notion")
async def control_notion() -> dict[str, int]:
    """Empty the fake workspace."""
    _notion_blocks.clear()
    _notion_children.clear()
    _notion_uploads.clear()
    return {"blocks": 0}


@app.get("/_control/notion/pages")
async def notion_pages() -> list[dict[str, Any]]:
    """Every page with its title, parent and the types of its top-level blocks, for itests."""
    return [
        {
            "id": block["id"],
            "title": block["child_page"]["title"],
            "parent": block["parent"],
            "blocks": [_notion_blocks[i]["type"] for i in _notion_children.get(block["id"], [])],
        }
        for block in _notion_blocks.values()
        if block["type"] == "child_page"
    ]


@app.post("/notion/v1/oauth/token", response_model=None)
async def notion_token(request: Request) -> dict[str, Any] | JSONResponse:
    if not request.headers.get("authorization", "").startswith("Basic "):
        return JSONResponse(status_code=401, content={"error": "invalid_client"})
    body = await request.json()
    if body.get("grant_type") == "authorization_code" and body.get("code") == BAD_OAUTH_CODE:
        return JSONResponse(status_code=400, content={"error": "invalid_grant"})
    return {
        "access_token": f"fake-notion-token-{uuid.uuid4().hex[:8]}",
        "token_type": "bearer",
        "refresh_token": f"fake-refresh-{uuid.uuid4().hex[:8]}",
        "bot_id": _new_id(),
        "workspace_id": _new_id(),
        "workspace_name": "Fake workspace",
        "workspace_icon": None,
        "owner": {"type": "workspace", "workspace": True},
        "duplicated_template_id": None,
    }


@app.post("/notion/v1/pages", response_model=None)
async def notion_create_page(request: Request) -> dict[str, Any] | JSONResponse:
    if not _signed_in(request):
        return _notion_error(401, "unauthorized", "API token is invalid.")
    body = await request.json()
    parent = (body.get("parent") or {}).get("page_id")
    title = "".join(
        part.get("text", {}).get("content", "")
        for part in body.get("properties", {}).get("title", {}).get("title", [])
    )
    if not parent:
        return _notion_error(400, "validation_error", "parent.page_id is required")
    page_id = _new_id()
    _notion_blocks[page_id] = {
        "object": "block",
        "id": page_id,
        "type": "child_page",
        "child_page": {"title": title},
        "parent": parent,
    }
    _notion_children.setdefault(parent, []).append(page_id)
    return {"object": "page", "id": page_id, "url": _page_url(page_id)}


@app.get("/notion/v1/pages/{page_id}", response_model=None)
async def notion_page(page_id: str, request: Request) -> dict[str, Any] | JSONResponse:
    if not _signed_in(request):
        return _notion_error(401, "unauthorized", "API token is invalid.")
    if page_id not in _notion_blocks:
        return _notion_error(404, "object_not_found", f"Could not find page {page_id}.")
    return {"object": "page", "id": page_id, "url": _page_url(page_id)}


@app.get("/notion/v1/blocks/{block_id}/children", response_model=None)
async def notion_children(block_id: str, request: Request) -> dict[str, Any] | JSONResponse:
    if not _signed_in(request):
        return _notion_error(401, "unauthorized", "API token is invalid.")
    ids = _notion_children.get(block_id, [])
    size = min(int(request.query_params.get("page_size", _NOTION_MAX_CHILDREN)), 100)
    start = int(request.query_params.get("start_cursor") or 0)
    end = start + size
    return {
        "object": "list",
        "results": [_public(_notion_blocks[i]) for i in ids[start:end]],
        "has_more": end < len(ids),
        "next_cursor": str(end) if end < len(ids) else None,
    }


@app.patch("/notion/v1/blocks/{block_id}/children", response_model=None)
async def notion_append(block_id: str, request: Request) -> dict[str, Any] | JSONResponse:
    if not _signed_in(request):
        return _notion_error(401, "unauthorized", "API token is invalid.")
    children = (await request.json()).get("children", [])
    if len(children) > _NOTION_MAX_CHILDREN:
        return _notion_error(400, "validation_error", "body.children.length should be ≤ 100")
    for child in children:
        upload = (child.get("image") or {}).get("file_upload")
        if upload and _notion_uploads.get(upload.get("id")) != "uploaded":
            return _notion_error(400, "validation_error", "file upload is not uploaded")
    results = [_store_block(block_id, child) for child in children]
    return {"object": "list", "results": results, "has_more": False, "next_cursor": None}


@app.post("/notion/v1/file_uploads", response_model=None)
async def notion_create_upload(request: Request) -> dict[str, Any] | JSONResponse:
    if not _signed_in(request):
        return _notion_error(401, "unauthorized", "API token is invalid.")
    upload_id = _new_id()
    _notion_uploads[upload_id] = "pending"
    return {"object": "file_upload", "id": upload_id, "status": "pending"}


@app.post("/notion/v1/file_uploads/{upload_id}/send", response_model=None)
async def notion_send_upload(upload_id: str, request: Request) -> dict[str, Any] | JSONResponse:
    if not _signed_in(request):
        return _notion_error(401, "unauthorized", "API token is invalid.")
    if _notion_uploads.get(upload_id) != "pending":
        return _notion_error(404, "object_not_found", f"Could not find file upload {upload_id}.")
    if not await request.body():  # multipart with the frame's bytes
        return _notion_error(400, "validation_error", "file is required")
    _notion_uploads[upload_id] = "uploaded"
    return {"object": "file_upload", "id": upload_id, "status": "uploaded"}


def _store_block(parent: str, child: dict[str, Any]) -> dict[str, Any]:
    block_id = _new_id()
    kind = child.get("type", "")
    nested = (child.get(kind) or {}).get("children", [])
    _notion_blocks[block_id] = {"object": "block", "id": block_id, "type": kind, "parent": parent}
    _notion_children.setdefault(parent, []).append(block_id)
    for grandchild in nested:
        _store_block(block_id, grandchild)
    return {"object": "block", "id": block_id}


def _public(block: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in block.items() if key != "parent"}


def _page_url(page_id: str) -> str:
    return f"https://www.notion.so/{page_id.replace('-', '')}"
