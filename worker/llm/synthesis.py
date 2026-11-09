"""Validate the LLM's JSON and apply the frame rules (ARCHITECTURE §8.2 steps 4-5).

Unparseable or schema-breaking output is an InvalidOutputError (the stage re-prompts, then
fails with LLM_INVALID_OUTPUT). Frames that break a rule are dropped and reported instead, so a
single bad pick never costs another LLM call: C-B needs every *kept* frame to be captioned.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from common.models import FramePick, Synthesis
from worker.llm.chunking import Chunk

_PROBLEM_MAX_CHARS = 300


class InvalidOutputError(Exception):
    """The answer is not usable at all; the message is fed back in the repair prompt."""


@dataclass
class Dropped:
    """Frames removed by the rules, as (t_sec, reason) for the log."""

    frames: list[tuple[Any, str]] = field(default_factory=list)

    def add(self, t_sec: Any, reason: str) -> None:
        self.frames.append((t_sec, reason))


def parse_part(text: str, chunk: Chunk, duration_sec: int, dropped: Dropped) -> Synthesis:
    """One chunk's answer → Synthesis whose frames lie inside the chunk and the video."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidOutputError(f"not valid JSON: {exc.msg} at char {exc.pos}") from exc
    if not isinstance(data, dict):
        raise InvalidOutputError("the answer is not a JSON object")
    data["frames"] = _usable_frames(data.get("frames"), dropped)
    try:
        part = Synthesis.model_validate(data)
    except ValidationError as exc:
        raise InvalidOutputError(_describe(exc)) from exc

    latest = min(chunk.end_sec, duration_sec - 1)
    kept: list[FramePick] = []
    for frame in part.frames:
        if not chunk.start_sec <= frame.t_sec <= latest:
            dropped.add(frame.t_sec, f"outside {chunk.start_sec}..{latest}s")
        else:
            kept.append(frame)
    return part.model_copy(update={"frames": kept})


def snap_to_candidates(
    part: Synthesis, candidates: Sequence[int], *, tolerance_s: int, dropped: Dropped
) -> Synthesis:
    """Every pick becomes the nearest candidate time (within `tolerance_s`) or is dropped, so a
    frame is always one the model was shown (D34); one pick per candidate."""
    kept: list[FramePick] = []
    used: set[int] = set()
    for frame in part.frames:
        nearest = min(candidates, key=lambda t: abs(t - frame.t_sec), default=None)
        if nearest is None or abs(nearest - frame.t_sec) > tolerance_s:
            dropped.add(frame.t_sec, "not a candidate frame")
        elif nearest in used:
            dropped.add(frame.t_sec, f"same candidate as {nearest}s")
        else:
            used.add(nearest)
            kept.append(frame.model_copy(update={"t_sec": nearest}))
    return part.model_copy(update={"frames": kept})


def finalize(
    synthesis: Synthesis, *, min_gap_s: int, max_frames: int, dropped: Dropped
) -> Synthesis:
    """Merged result → frames in time order, at least `min_gap_s` apart, at most `max_frames`."""
    kept: list[FramePick] = []
    for frame in sorted(synthesis.frames, key=lambda f: f.t_sec):
        if kept and frame.t_sec - kept[-1].t_sec < min_gap_s:
            dropped.add(frame.t_sec, f"within {min_gap_s}s of {kept[-1].t_sec}s")
        elif len(kept) >= max_frames:
            dropped.add(frame.t_sec, f"over MAX_FRAMES={max_frames}")
        else:
            kept.append(frame)
    sections = [s for s in synthesis.sections if s.bullets]
    return synthesis.model_copy(update={"frames": kept, "sections": sections})


def _usable_frames(raw: Any, dropped: Dropped) -> list[dict[str, Any]]:
    """Frames with an integer time and a non-empty caption and reason; others dropped."""
    if not isinstance(raw, list):
        return raw  # type: ignore[no-any-return]  # schema validation reports it
    usable: list[dict[str, Any]] = []
    for item in raw:
        t_sec = item.get("t_sec") if isinstance(item, dict) else None
        if not isinstance(item, dict):
            dropped.add(None, "not an object")
        elif not isinstance(t_sec, int | float) or isinstance(t_sec, bool) or t_sec < 0:
            dropped.add(t_sec, "no valid t_sec")
        elif not _text(item.get("caption")) or not _text(item.get("reason")):
            dropped.add(t_sec, "missing caption or reason")  # C-B: no uncaptioned frame
        else:
            usable.append({**item, "t_sec": int(t_sec)})
    return usable


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _describe(exc: ValidationError) -> str:
    problems = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
    return "; ".join(problems)[:_PROBLEM_MAX_CHARS]
