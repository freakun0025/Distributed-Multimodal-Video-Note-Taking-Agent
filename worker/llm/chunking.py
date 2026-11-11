"""Transcript chunking and in-order merge of partial syntheses (ARCHITECTURE §8.2)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from common.models import Question, Section, Synthesis, TranscriptSegment

# Each chunk repeats the end of the previous one, so a point made across the boundary is seen
# whole by at least one call.
OVERLAP_FRACTION = 0.05


def clock(seconds: float) -> str:
    """`[h:]mm:ss` for a position in the video."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


@dataclass(frozen=True)
class Chunk:
    index: int
    start_sec: int  # first line's start, rounded down
    end_sec: int  # last line's end, rounded up
    lines: tuple[str, ...]
    starts: tuple[float, ...] = ()  # each line's start in seconds (to place candidate frames)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True)
class _Line:
    start: float
    end: float
    text: str


def chunk_transcript(segments: Sequence[TranscriptSegment], max_chars: int) -> list[Chunk]:
    """Split on segment boundaries into chunks of at most `max_chars` (a longer single line
    becomes its own chunk), each starting with a small overlap from the previous one."""
    lines = [
        _Line(seg.start, seg.start + seg.duration, f"[{clock(seg.start)}] {seg.text}")
        for seg in segments
    ]
    overlap_chars = int(max_chars * OVERLAP_FRACTION)
    groups: list[list[_Line]] = []
    current: list[_Line] = []
    size = 0
    for line in lines:
        cost = len(line.text) + 1
        if current and size + cost > max_chars:
            groups.append(current)
            current = _tail(current, overlap_chars)
            size = sum(len(item.text) + 1 for item in current)
            if size + cost > max_chars:  # overlap would not leave room: drop it
                current, size = [], 0
        current.append(line)
        size += cost
    if current:
        groups.append(current)
    return [
        Chunk(
            index=i,
            start_sec=math.floor(group[0].start),
            end_sec=math.ceil(max(line.end for line in group)),
            lines=tuple(line.text for line in group),
            starts=tuple(line.start for line in group),
        )
        for i, group in enumerate(groups)
    ]


def _tail(lines: list[_Line], max_chars: int) -> list[_Line]:
    tail: list[_Line] = []
    size = 0
    for line in reversed(lines):
        size += len(line.text) + 1
        if size > max_chars:
            break
        tail.insert(0, line)
    return tail


def merge(parts: Sequence[Synthesis]) -> Synthesis:
    """Chunk results in chunk order: sections concatenated (a heading continued across a
    boundary is merged and keeps its first start), summaries joined, questions concatenated
    without repeats, frames sorted by time."""
    sections: list[Section] = []
    for part in parts:
        for section in part.sections:
            previous = sections[-1] if sections else None
            if previous and previous.heading.casefold() == section.heading.casefold():
                bullets = previous.bullets + [
                    b for b in section.bullets if b not in previous.bullets
                ]
                sections[-1] = previous.model_copy(update={"bullets": bullets})
            else:
                if previous and section.start_sec < previous.start_sec:  # overlap at the seam
                    section = section.model_copy(update={"start_sec": previous.start_sec})
                sections.append(section)
    questions: list[Question] = []
    asked: set[str] = set()
    for part in parts:
        for question in part.questions:
            if (key := question.question.strip().casefold()) not in asked:
                asked.add(key)
                questions.append(question)
    return Synthesis(
        title=parts[0].title,
        summary=" ".join(part.summary.strip() for part in parts if part.summary.strip()),
        sections=sections,
        questions=questions,
        frames=sorted((f for part in parts for f in part.frames), key=lambda f: f.t_sec),
    )
