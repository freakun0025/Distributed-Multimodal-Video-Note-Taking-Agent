"""LLM output validation, frame rules and the prompt/schema (ARCHITECTURE §8.2 steps 3-4)."""

import json
from typing import Any

import pytest

from common.models import FramePick, Section, Synthesis, VideoMeta
from worker.llm.chunking import Chunk
from worker.llm.prompts import synthesis_request, synthesis_schema
from worker.llm.synthesis import Dropped, InvalidOutputError, finalize, parse_part

CHUNK = Chunk(index=0, start_sec=0, end_sec=300, lines=("[0:00] hello",))
FRAME = {"t_sec": 61, "caption": "A diagram of Raft.", "reason": "Shows leader election."}


def _answer(**overrides: Any) -> str:
    data = {
        "title": "Raft",
        "summary": "Consensus.",
        "sections": [{"heading": "Leaders", "bullets": ["one leader per term"]}],
        "frames": [FRAME],
    }
    data.update(overrides)
    return json.dumps(data)


def test_a_valid_answer_parses() -> None:
    part = parse_part(_answer(), CHUNK, duration_sec=600, dropped=Dropped())
    assert part.frames == [FramePick.model_validate(FRAME)]
    assert part.sections == [Section(heading="Leaders", bullets=["one leader per term"])]


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ('{"title": "Raft", ', "not valid JSON"),
        ("[1, 2]", "not a JSON object"),
        (json.dumps({"title": "x", "summary": "y", "frames": []}), "sections"),
        (_answer(sections=[{"heading": "", "bullets": []}]), "heading"),
    ],
)
def test_unusable_answers_raise_with_a_problem_for_the_repair_prompt(
    text: str, problem: str
) -> None:
    with pytest.raises(InvalidOutputError, match=problem):
        parse_part(text, CHUNK, duration_sec=600, dropped=Dropped())


def test_bad_frames_are_dropped_not_repaired() -> None:
    dropped = Dropped()
    frames = [
        FRAME,
        {**FRAME, "t_sec": 90, "caption": "  "},  # C-B: a frame needs a caption
        {**FRAME, "t_sec": 95, "reason": None},
        {**FRAME, "t_sec": -1},
        {**FRAME, "t_sec": True},
        {**FRAME, "t_sec": 120.7},  # fractional seconds: whole seconds kept
        {**FRAME, "t_sec": 301},  # after this chunk
        "not an object",
    ]
    part = parse_part(_answer(frames=frames), CHUNK, duration_sec=600, dropped=dropped)
    assert [f.t_sec for f in part.frames] == [61, 120]
    assert len(dropped.frames) == 6


def test_frames_past_the_end_of_the_video_are_dropped() -> None:
    dropped = Dropped()
    part = parse_part(
        _answer(frames=[{**FRAME, "t_sec": 250}]), CHUNK, duration_sec=250, dropped=dropped
    )
    assert part.frames == []
    assert dropped.frames == [(250, "outside 0..249s")]


def test_finalize_orders_spaces_and_caps_frames_and_drops_empty_sections() -> None:
    synthesis = Synthesis(
        title="t",
        summary="s",
        sections=[Section(heading="Empty", bullets=[]), Section(heading="Kept", bullets=["b"])],
        frames=[FramePick(t_sec=t, caption="c", reason="r") for t in (100, 5, 12, 40, 70, 300)],
    )
    dropped = Dropped()
    final = finalize(synthesis, min_gap_s=10, max_frames=3, dropped=dropped)
    assert [f.t_sec for f in final.frames] == [5, 40, 70]
    assert [reason for _, reason in dropped.frames] == [
        "within 10s of 5s",
        "over MAX_FRAMES=3",
        "over MAX_FRAMES=3",
    ]
    assert [s.heading for s in final.sections] == ["Kept"]


def test_schema_is_self_contained_and_bounds_the_frame_count() -> None:
    schema = synthesis_schema(2, 4)
    assert "$ref" not in json.dumps(schema)
    assert "$defs" not in schema
    frames = schema["properties"]["frames"]
    assert (frames["minItems"], frames["maxItems"]) == (2, 4)
    frame = schema["properties"]["frames"]["items"]
    assert set(frame["required"]) == {"t_sec", "caption", "reason"}


def test_the_request_states_language_frame_budget_and_time_window() -> None:
    meta = VideoMeta(video_id="v", title="Distributed Systems 6", duration_sec=1200)
    chunk = Chunk(index=1, start_sec=590, end_sec=1200, lines=("[9:50] so the leader",))
    request = synthesis_request(meta, chunk, 2, language="Hindi", min_frames=2, max_frames=4)
    assert "in Hindi" in request.system
    assert "between 2 and 4 moments" in request.system
    assert "pick none" not in request.system  # every video gets frames (ARCHITECTURE §8.2: 3-8)
    assert "between 590 and 1200" in request.system
    assert "Part 2 of 2, from 9:50 to 20:00" in request.prompt
    assert "[9:50] so the leader" in request.prompt
    assert "could not be used" not in request.prompt

    repair = synthesis_request(
        meta, chunk, 2, language="Hindi", min_frames=2, max_frames=4, problem="bad JSON"
    )
    assert "could not be used (bad JSON)" in repair.prompt
