"""How much the LLM must write and pick for a part of a lecture, and where (CP7c, D35).

The floors go into the response schema as `minItems`, which Gemini enforces while decoding, so
even the fallback model cannot hand back a one-paragraph abstract for an 18-minute lecture. Each
counted item enlarges the schema Gemini compiles, so the floors are capped; the prompt asks for
more wherever the lecture covers more.

Frame slots do the same for coverage: the first frames are `prefixItems`, frame i limited to the
candidates of stretch i (`enum`), so the pictures span the whole part instead of bunching early.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from common.config import Settings


@dataclass(frozen=True)
class NotesFloor:
    sections: int  # least sections in the part
    bullets: int  # least bullets in each section


NO_FLOOR = NotesFloor(sections=1, bullets=1)


def notes_floor(span_s: int, settings: Settings) -> NotesFloor:
    """One section per NOTES_SECTION_EVERY_S of the part and NOTES_BULLETS_PER_MINUTE bullets
    per minute spread over those sections; each at least 1 and at most its cap."""
    span = max(span_s, 0)
    sections = min(
        max(1, span // settings.notes_section_every_s), settings.notes_floor_max_sections
    )
    total_bullets = span / 60 * settings.notes_bullets_per_minute
    bullets = min(max(1, math.ceil(total_bullets / sections)), settings.notes_floor_max_bullets)
    return NotesFloor(sections=sections, bullets=bullets)


def min_frames_for(duration_s: int, settings: Settings) -> int:
    """At least one frame per MIN_FRAME_EVERY_S of video, within MIN_FRAMES..MAX_FRAMES."""
    wanted = max(duration_s, 0) // settings.min_frame_every_s
    return min(max(settings.min_frames, wanted), settings.max_frames)


@dataclass(frozen=True)
class FrameSlot:
    """One stretch of a part that must supply a frame: any whole second in `first..last`, or
    with candidate frames, exactly one of `choices` (all inside the stretch)."""

    first: int
    last: int
    choices: tuple[int, ...] = ()


def frame_slots(
    first: int, last: int, count: int, candidates: Sequence[int] = ()
) -> list[FrameSlot]:
    """`count` equal stretches of `first..last` (whole seconds, inclusive), one frame from each,
    so the required frames cover the part from start to end. With candidates, a stretch that has
    none is left out rather than forcing a pick from elsewhere."""
    span = last - first + 1
    if count <= 0 or span <= 0:
        return []
    slots: list[FrameSlot] = []
    for i in range(count):
        lo = first + (i * span) // count
        hi = first + ((i + 1) * span) // count - 1
        if hi < lo:
            continue
        if candidates:
            inside = tuple(sorted(t for t in set(candidates) if lo <= t <= hi))
            if inside:
                slots.append(FrameSlot(lo, hi, inside))
        else:
            slots.append(FrameSlot(lo, hi))
    return slots
