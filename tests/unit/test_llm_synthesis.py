"""LLM output validation, frame rules and the prompt/schema (ARCHITECTURE §8.2 steps 3-4)."""

import json
from typing import Any

import pytest

from common.models import FramePick, Question, Section, Synthesis, VideoMeta
from worker.llm.base import PromptImage
from worker.llm.budget import FrameSlot, NotesFloor, QuestionBudget
from worker.llm.chunking import Chunk
from worker.llm.prompts import CandidateImage, synthesis_request, synthesis_schema
from worker.llm.synthesis import (
    Dropped,
    InvalidOutputError,
    finalize,
    parse_part,
    snap_to_candidates,
)

CHUNK = Chunk(index=0, start_sec=0, end_sec=300, lines=("[0:00] hello",))
FRAME = {"t_sec": 61, "caption": "A diagram of Raft.", "reason": "Shows leader election."}
QUESTION = {"question": "Why does Raft elect a leader?", "answer": "So one node orders writes."}


def _answer(**overrides: Any) -> str:
    data = {
        "title": "Raft",
        "summary": "Consensus.",
        "sections": [{"heading": "Leaders", "start": "0:00", "bullets": ["one leader per term"]}],
        "questions": [QUESTION],
        "frames": [FRAME],
    }
    data.update(overrides)
    return json.dumps(data)


def test_a_valid_answer_parses() -> None:
    part = parse_part(_answer(), CHUNK, duration_sec=600, dropped=Dropped())
    assert part.frames == [FramePick.model_validate(FRAME)]
    assert part.sections == [
        Section(heading="Leaders", start_sec=0, bullets=["one leader per term"])
    ]
    assert part.questions == [Question.model_validate(QUESTION)]


def test_section_starts_are_kept_inside_the_part_and_in_order() -> None:
    chunk = Chunk(index=1, start_sec=300, end_sec=600, lines=("[5:00] hello",))
    sections = [
        {"heading": h, "start": t, "bullets": ["b"]}
        for h, t in (("A", "4:10"), ("B", "[7:00]"), ("C", "6:40"), ("D", "15:00"))
    ]
    part = parse_part(_answer(sections=sections), chunk, duration_sec=500, dropped=Dropped())
    # 4:10 is before the part, 6:40 before the section above it, 15:00 after the video's end.
    assert [s.start_sec for s in part.sections] == [300, 420, 420, 499]


def test_section_starts_are_copied_times_and_need_no_arithmetic_from_the_model() -> None:
    # D38: asked for seconds, Flash-Lite turned [9:41] into 941; now it copies "9:41".
    chunk = Chunk(index=0, start_sec=0, end_sec=4000, lines=("[0:00] hi",))
    sections = [
        {"heading": h, "start": t, "bullets": ["b"]}
        for h, t in (("A", "0:00"), ("B", "9:41"), ("C", "941"), ("D", None), ("E", "1:02:03"))
    ]
    dropped = Dropped()
    part = parse_part(_answer(sections=sections), chunk, duration_sec=4000, dropped=dropped)
    # A bare number or no start at all: the previous section's start, reported, no repair.
    assert [s.start_sec for s in part.sections] == [0, 581, 581, 581, 3723]
    assert dropped.starts == ["941", None]


def test_blank_questions_are_left_out_without_a_repair() -> None:
    questions = [QUESTION, {"question": " ", "answer": "x"}, {"question": "Why?", "answer": ""}]
    part = parse_part(_answer(questions=questions), CHUNK, duration_sec=600, dropped=Dropped())
    assert part.questions == [Question.model_validate(QUESTION)]


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ('{"title": "Raft", ', "not valid JSON"),
        ("[1, 2]", "not a JSON object"),
        (json.dumps({"title": "x", "summary": "y", "frames": []}), "sections"),
        (_answer(sections=[{"heading": "", "start": "0:00", "bullets": []}]), "heading"),
        (_answer(questions=None), "questions"),
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
        sections=[
            Section(heading="Empty", start_sec=0, bullets=[]),
            Section(heading="Kept", start_sec=0, bullets=["b"]),
        ],
        questions=[Question(question=f"Q{i}?", answer="A.") for i in range(10)],
        frames=[FramePick(t_sec=t, caption="c", reason="r") for t in (100, 5, 12, 40, 70, 300)],
    )
    dropped = Dropped()
    final = finalize(synthesis, min_gap_s=10, max_frames=3, max_questions=8, dropped=dropped)
    assert [q.question for q in final.questions] == [f"Q{i}?" for i in range(8)]
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


def test_schema_enforces_the_notes_floor_on_sections_and_their_bullets() -> None:
    schema = synthesis_schema(2, 4, NotesFloor(sections=6, bullets=5))
    sections = schema["properties"]["sections"]
    assert sections["minItems"] == 6
    assert sections["items"]["properties"]["bullets"]["minItems"] == 5
    assert "maxItems" not in sections  # a floor, never a ceiling on depth


def test_schema_bounds_the_questions_and_asks_for_section_starts_as_text() -> None:
    schema = synthesis_schema(2, 4, questions=QuestionBudget(least=5, most=8))
    questions = schema["properties"]["questions"]
    assert (questions["minItems"], questions["maxItems"]) == (5, 8)
    assert set(questions["items"]["required"]) == {"question", "answer"}
    section = schema["properties"]["sections"]["items"]
    assert list(section["properties"]) == ["heading", "start", "bullets"]
    assert section["properties"]["start"]["type"] == "string"
    assert set(section["required"]) == {"heading", "start", "bullets"}
    order = list(schema["properties"])  # Gemini writes the properties in schema order
    assert order.index("sections") < order.index("questions") < order.index("frames")


def test_the_prompt_asks_for_revision_depth_with_a_short_recap() -> None:
    meta = VideoMeta(video_id="v", title="Trees", duration_sec=1088)
    chunk = Chunk(index=0, start_sec=0, end_sec=1088, lines=("[0:00] hi",), starts=(0.0,))
    request = synthesis_request(
        meta,
        chunk,
        1,
        language="English",
        min_frames=6,
        max_frames=8,
        floor=NotesFloor(sections=6, bullets=5),
    )
    system = request.system
    assert "revise this material from your notes instead of watching the video again" in system
    assert "summary: 2-4 sentences" in system
    assert "at least 6, more if the part covers more topics" in system
    assert "at least 5 in each section, and as many more as the topic needs" in system
    for must in ("definition", "formula", "ordered steps", "actual numbers", "pitfalls"):
        assert must in system
    assert "concise" not in system  # the CP7b wording that produced 9-bullet notes
    assert request.schema["properties"]["sections"]["minItems"] == 6


def test_the_prompt_asks_for_complete_notes_without_padding() -> None:
    # D37: Gemini 3 answers briefly unless asked for detail, and wrote exactly the floor with
    # whole topics dropped; a higher floor then drew commentary the speaker never made.
    meta = VideoMeta(video_id="v", title="Trees", duration_sec=600)
    chunk = Chunk(index=0, start_sec=0, end_sec=600, lines=("[0:00] hi",), starts=(0.0,))
    system = synthesis_request(
        meta, chunk, 1, language="English", min_frames=3, max_frames=8
    ).system
    assert "Every teaching point of the part goes into the notes" in system
    assert "Shorten the wording, never the content" in system
    assert "Longer, more detailed notes are better" in system
    assert "Never pad: every bullet adds a new point" in system
    assert "the final topics get as much detail as the first ones" in system
    assert 'Start a definition with the term and a colon ("Photosynthesis: ...")' in system
    assert "common mistakes the speaker gives" in system  # never an invented pitfall
    for tag in ("role", "completeness", "content", "format", "questions", "frames", "output"):
        assert system.count(f"<{tag}>") == system.count(f"</{tag}>") == 1


def test_the_prompt_keeps_caveats_and_adds_no_examples_or_detail_of_its_own() -> None:
    # D38, from the Claude Code tutorial: "Claude 3.5 Sonnet" where the video named no model,
    # a survey the speaker called "not a super rigorous study" turned into "an empirical
    # analysis", and "event listeners and CSS" where he said only "change the code".
    meta = VideoMeta(video_id="v", title="Apps", duration_sec=600)
    chunk = Chunk(index=0, start_sec=0, end_sec=600, lines=("[0:00] hi",), starts=(0.0,))
    system = synthesis_request(
        meta, chunk, 1, language="English", min_frames=3, max_frames=8
    ).system
    assert "the limits the speaker puts on a claim (only a survey, a small sample" in system
    assert "kept in the same bullet as the claim itself" in system
    assert "though the speaker notes it was only Z" in system
    assert 'never add your own, not even after "such as" or "for example"' in system
    assert "it cannot go out of date (never a product version" in system
    assert "at the level of detail the video gives" in system


def test_the_prompt_asks_for_section_starts_and_self_test_questions() -> None:
    meta = VideoMeta(video_id="v", title="Apps", duration_sec=1200)
    chunk = Chunk(index=1, start_sec=590, end_sec=1200, lines=("[9:50] so",), starts=(590.0,))
    request = synthesis_request(
        meta,
        chunk,
        2,
        language="English",
        min_frames=3,
        max_frames=8,
        questions=QuestionBudget(least=3, most=4),
    )
    assert "start of a section: the [minutes:seconds] time of the transcript line" in (
        request.system
    )
    assert '(for example "9:47"); between 9:50 and 20:00' in request.system
    assert "questions: 3 to 4 self-test questions" in request.system
    assert "Never ask anything the video does not answer" in request.system
    assert request.schema["properties"]["questions"]["minItems"] == 3


def test_the_prompt_fits_any_subject_and_asks_for_specifics_only_where_present() -> None:
    meta = VideoMeta(video_id="v", title="The Congress of Vienna", duration_sec=600)
    chunk = Chunk(index=0, start_sec=0, end_sec=600, lines=("[0:00] hi",), starts=(0.0,))
    system = synthesis_request(
        meta, chunk, 1, language="English", min_frames=3, max_frames=8
    ).system
    always, _, when_present = system.partition("Whenever the part contains them")
    for core in ("key concept", "how and why: the mechanism", "trade-offs", "takeaways"):
        assert core in always
    for specific in ("formulas", "actual numbers", "code", "dates", "order of events"):
        assert specific in when_present
    assert "never make one up" in when_present
    assert "lecture or tutorial" in system
    assert "history, the humanities" in system
    assert "Leave out sponsor messages, channel promotion" in system  # a Patreon bullet, CP7c
    # The video is the source, but certain standard background may be added (a history lecture
    # got correct dates the speaker left out, which the user wants); no subject-specific ban.
    assert "The video is the source" in system
    assert "never contradict it" in system
    assert "only if you are certain it is correct" in system
    assert "add no outside facts" not in system
    assert "exam-revision" not in system  # not only for exam subjects


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
    # The task is restated after the transcript (Google's guide: instructions after a long
    # context), so the prompt ends with it.
    assert request.prompt.endswith(
        "Based on the transcript above, write the study notes for part 2 of 2 (9:50 to 20:00), "
        "following the rules: every teaching point, in the video's order, in full detail, "
        "then the self-test questions."
    )

    repair = synthesis_request(
        meta, chunk, 2, language="Hindi", min_frames=2, max_frames=4, problem="bad JSON"
    )
    assert "could not be used (bad JSON)" in repair.prompt
    assert repair.prompt.index("questions.") < repair.prompt.index("could not be used")


def test_picks_snap_to_the_candidate_frames_shown_or_are_dropped() -> None:
    part = Synthesis(
        title="t",
        summary="s",
        sections=[],
        questions=[],
        frames=[FramePick(t_sec=t, caption="c", reason="r") for t in (45, 106, 107, 200)],
    )
    dropped = Dropped()
    snapped = snap_to_candidates(part, [45, 105, 150], tolerance_s=2, dropped=dropped)
    assert [f.t_sec for f in snapped.frames] == [45, 105]
    assert dropped.frames == [(107, "same candidate as 105s"), (200, "not a candidate frame")]


def test_candidate_frames_sit_in_the_transcript_at_their_time() -> None:
    meta = VideoMeta(video_id="v", title="Raft", duration_sec=120)
    chunk = Chunk(
        index=0,
        start_sec=0,
        end_sec=120,
        lines=("[0:00] intro", "[0:40] the leader", "[1:10] votes"),
        starts=(0.0, 40.0, 70.0),
    )
    images = [CandidateImage(t_sec=45, data=b"B"), CandidateImage(t_sec=15, data=b"A")]
    request = synthesis_request(
        meta, chunk, 1, language="English", min_frames=1, max_frames=2, candidates=images
    )
    order = [p.data.decode() if isinstance(p, PromptImage) else p for p in request.parts]
    assert order[1:-1] == ["A", "\n[0:40] the leader\n[0:45] Candidate frame t_sec=45:", "B"]
    assert order[-1].startswith("\n[1:10] votes\n\nBased on the transcript and frames above")
    assert order[0].endswith("[0:00] intro\n[0:15] Candidate frame t_sec=15:")
    assert request.images == (PromptImage(data=b"A"), PromptImage(data=b"B"))
    assert 'labelled "Candidate frame t_sec=N"' in request.system
    assert "between 1 and 2 of these candidates" in request.system
    assert "only have the transcript" not in request.system
    # CP7c frame rules: whole timeline, complete slides, captions of what is visible only.
    assert "from its first minutes to its last" in request.system
    assert "at least one for every section" in request.system
    assert "empty, greyed out or still being built" in request.system
    assert "never what is said around it" in request.system
    assert "a frame of only the speaker" in request.system  # last resort, like a title card
    # D38: "Sonnet 5" on screen came back as "Sonnet 3.5": copy exactly or leave it out.
    assert "leave out any you cannot read with certainty" in request.system


def test_without_candidates_the_prompt_is_text_only() -> None:
    meta = VideoMeta(video_id="v", title="Raft", duration_sec=120)
    chunk = Chunk(index=0, start_sec=0, end_sec=120, lines=("[0:00] intro",), starts=(0.0,))
    request = synthesis_request(meta, chunk, 1, language="English", min_frames=3, max_frames=8)
    assert len(request.parts) == 1 and request.images == ()
    assert "only have the transcript" in request.system
    assert request.prompt.endswith("in full detail, then the self-test questions.")
    assert "Based on the transcript above" in request.prompt


def test_slots_make_frame_i_come_from_stretch_i_and_only_from_its_candidates() -> None:
    slots = [FrameSlot(0, 59, (15, 45)), FrameSlot(60, 119, (105,))]
    schema = synthesis_schema(3, 4, slots=slots)
    frames = schema["properties"]["frames"]
    assert (frames["minItems"], frames["maxItems"]) == (2, 4)  # one per slot
    allowed = [item["properties"]["t_sec"] for item in frames["prefixItems"]]
    assert allowed == [{"type": "integer", "enum": [15, 45]}, {"type": "integer", "enum": [105]}]
    assert frames["items"]["properties"]["t_sec"] == {"type": "integer", "enum": [15, 45, 105]}
    assert set(frames["prefixItems"][0]["required"]) == {"t_sec", "caption", "reason"}


def test_slots_without_candidates_bound_each_frame_to_its_seconds() -> None:
    schema = synthesis_schema(2, 8, slots=[FrameSlot(0, 299), FrameSlot(300, 599)])
    frames = schema["properties"]["frames"]
    assert [item["properties"]["t_sec"] for item in frames["prefixItems"]] == [
        {"type": "integer", "minimum": 0, "maximum": 299},
        {"type": "integer", "minimum": 300, "maximum": 599},
    ]
    assert frames["items"]["properties"]["t_sec"]["maximum"] == 599


def test_the_prompt_explains_the_slots_only_when_there_are_slots() -> None:
    meta = VideoMeta(video_id="v", title="Trees", duration_sec=600)
    chunk = Chunk(index=0, start_sec=0, end_sec=600, lines=("[0:00] hi",), starts=(0.0,))
    with_slots = synthesis_request(
        meta,
        chunk,
        1,
        language="English",
        min_frames=3,
        max_frames=8,
        slots=[FrameSlot(0, 299), FrameSlot(300, 599)],
    )
    assert "Frames 1 to 2 each come from their own stretch" in with_slots.system
    assert "between 2 and 8 moments" in with_slots.system  # min = number of slots
    plain = synthesis_request(meta, chunk, 1, language="English", min_frames=3, max_frames=8)
    assert "own stretch" not in plain.system
    assert "\n\n-" not in plain.system  # no blank line left where the rule would be
