"""A job's Notion page, built deterministically from its artefacts (ARCHITECTURE §8.4).

The same synthesis.json and frames/manifest.json always give the same block list, so a publish
that stopped part-way appends blocks[k:], where k is the number of blocks already on the page.
Layout (user's CP8 decisions, D38): a video block (a playable thumbnail that links back to
YouTube) -> a callout with the link and duration -> the summary -> for each section a heading
whose timestamp links to that moment, its bullets (flat), then the frames whose time falls in
the section's range -> "Test yourself" with one toggle per question, the answer inside.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from common.models import Question, Section, Synthesis, VideoMeta
from worker.llm.chunking import clock

RICH_TEXT_MAX = 2000  # characters per rich text object (Notion request limits)
QUESTIONS_HEADING = "Test yourself"
PENDING_UPLOAD = ""  # an image block's upload ID until the frame is uploaded, just before append


@dataclass(frozen=True)
class PageFrame:
    """A frame that was saved (frames/manifest.json) with the caption the notes gave it."""

    t_sec: int
    path: Path
    caption: str
    reason: str


@dataclass(frozen=True)
class PageBlocks:
    blocks: list[dict[str, Any]]
    images: dict[int, PageFrame]  # block index -> the frame its image block shows


def job_tag(job_id: UUID) -> str:
    """Ends every page title, so a retried publish finds its page instead of making another."""
    return f" · #{str(job_id)[:8]}"


def page_title(video_title: str, job_id: UUID) -> str:
    return f"{video_title.strip() or 'Video notes'}{job_tag(job_id)}"


def at(watch_url: str, t_sec: int) -> str:
    """The video at a moment."""
    return f"{watch_url}&t={t_sec}s"


def build_page(
    meta: VideoMeta, synthesis: Synthesis, frames: Sequence[PageFrame], *, watch_url: str
) -> PageBlocks:
    blocks: list[dict[str, Any]] = [
        _block("video", {"type": "external", "external": {"url": watch_url}}),
        _callout(meta, watch_url),
    ]
    if synthesis.summary.strip():
        blocks.append(_block("paragraph", {"rich_text": rich_text(synthesis.summary)}))
    images: dict[int, PageFrame] = {}

    def add_frames(placed: Sequence[PageFrame]) -> None:
        for frame in placed:
            images[len(blocks)] = frame
            blocks.append(_image(frame, watch_url))

    if not synthesis.sections:
        add_frames(sorted(frames, key=lambda f: f.t_sec))
    for section, placed in zip(
        synthesis.sections, place_frames(synthesis.sections, frames), strict=True
    ):
        blocks.append(_heading(section, watch_url))
        blocks.extend(
            _block("bulleted_list_item", {"rich_text": rich_text(bullet)})
            for bullet in section.bullets
        )
        add_frames(placed)
    if synthesis.questions:
        blocks.append(_block("heading_2", {"rich_text": rich_text(QUESTIONS_HEADING)}))
        blocks.extend(_toggle(question) for question in synthesis.questions)
    return PageBlocks(blocks=blocks, images=images)


def place_frames(sections: Sequence[Section], frames: Sequence[PageFrame]) -> list[list[PageFrame]]:
    """Each frame goes after the section whose range [start_sec, next start_sec) holds it; a
    frame before the first section's start goes with the first section."""
    placed: list[list[PageFrame]] = [[] for _ in sections]
    if not sections:
        return placed
    starts = [section.start_sec for section in sections]
    for frame in sorted(frames, key=lambda f: f.t_sec):
        placed[max(bisect.bisect_right(starts, frame.t_sec) - 1, 0)].append(frame)
    return placed


def with_upload(block: dict[str, Any], upload_id: str) -> dict[str, Any]:
    """An image block from build_page with its file upload attached."""
    image = {**block["image"], "file_upload": {"id": upload_id}}
    return {**block, "image": image}


def rich_text(content: str, *, link: str | None = None) -> list[dict[str, Any]]:
    """Text as rich text objects of at most RICH_TEXT_MAX characters each."""
    pieces = [content[i : i + RICH_TEXT_MAX] for i in range(0, len(content), RICH_TEXT_MAX)]
    return [
        {"type": "text", "text": {"content": piece, "link": {"url": link} if link else None}}
        for piece in pieces or [""]
    ]


def _block(kind: str, body: dict[str, Any]) -> dict[str, Any]:
    return {"object": "block", "type": kind, kind: body}


def _callout(meta: VideoMeta, watch_url: str) -> dict[str, Any]:
    details = " · ".join(part for part in (clock(meta.duration_sec), meta.channel) if part)
    return _block(
        "callout",
        {
            "rich_text": rich_text("Watch on YouTube", link=watch_url) + rich_text(f" · {details}"),
            "icon": {"type": "emoji", "emoji": "🎬"},
        },
    )


def _heading(section: Section, watch_url: str) -> dict[str, Any]:
    text = f"{clock(section.start_sec)} · {section.heading}"
    return _block(
        "heading_2", {"rich_text": rich_text(text, link=at(watch_url, section.start_sec))}
    )


def _image(frame: PageFrame, watch_url: str) -> dict[str, Any]:
    caption = rich_text(f"{frame.caption} — why: {frame.reason} ") + rich_text(
        f"({clock(frame.t_sec)})", link=at(watch_url, frame.t_sec)
    )
    return _block(
        "image",
        {"type": "file_upload", "file_upload": {"id": PENDING_UPLOAD}, "caption": caption},
    )


def _toggle(question: Question) -> dict[str, Any]:
    answer = _block("paragraph", {"rich_text": rich_text(question.answer)})
    return _block("toggle", {"rich_text": rich_text(question.question), "children": [answer]})
