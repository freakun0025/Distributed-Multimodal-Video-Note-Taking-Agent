"""Transcript chunking and in-order merge (ARCHITECTURE §8.2 steps 2 and 5)."""

import itertools
from collections.abc import Sequence

from common.models import FramePick, Question, Section, Synthesis, TranscriptSegment
from worker.llm.chunking import chunk_transcript, clock, merge


def _segments(count: int, every: float = 10.0, text: str = "x" * 40) -> list[TranscriptSegment]:
    return [TranscriptSegment(start=i * every, duration=every, text=text) for i in range(count)]


def test_clock_formats_minutes_and_hours() -> None:
    assert [clock(0), clock(61.9), clock(1200), clock(3725)] == ["0:00", "1:01", "20:00", "1:02:05"]


def test_a_short_transcript_is_one_chunk_with_its_time_range() -> None:
    (chunk,) = chunk_transcript(_segments(5), max_chars=10_000)
    assert (chunk.index, chunk.start_sec, chunk.end_sec) == (0, 0, 50)
    assert chunk.lines[1] == "[0:10] " + "x" * 40
    assert chunk.text.count("\n") == 4


def test_long_transcripts_split_on_segment_boundaries_with_overlap() -> None:
    segments = _segments(60)  # 49 chars per line incl. newline
    chunks = chunk_transcript(segments, max_chars=1000)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 for c in chunks)
    for earlier, later in itertools.pairwise(chunks):
        assert later.lines[0] in earlier.lines  # overlap: the next chunk repeats the end
        assert later.start_sec < earlier.end_sec
    # Every segment appears, in order.
    seen = [line for chunk in chunks for line in chunk.lines]
    assert sorted(set(seen), key=seen.index) == [f"[{clock(s.start)}] {s.text}" for s in segments]
    assert chunks[-1].end_sec == 600


def test_a_single_line_longer_than_the_limit_is_its_own_chunk() -> None:
    segments = [*_segments(2), TranscriptSegment(start=20, duration=5, text="y" * 500)]
    chunks = chunk_transcript(segments, max_chars=200)
    assert chunks[-1].lines == ("[0:20] " + "y" * 500,)


def _part(
    sections: list[tuple[str, list[str]]],
    frames: list[int],
    summary: str,
    *,
    starts: list[int] | None = None,
    questions: Sequence[str] = (),
) -> Synthesis:
    starts = starts or [0] * len(sections)
    return Synthesis(
        title=f"Notes {summary}",
        summary=summary,
        sections=[
            Section(heading=h, start_sec=t, bullets=b)
            for (h, b), t in zip(sections, starts, strict=True)
        ],
        questions=[Question(question=q, answer=f"A: {q}") for q in questions],
        frames=[FramePick(t_sec=t, caption=f"c{t}", reason="r") for t in frames],
    )


def test_merge_keeps_chunk_order_and_joins_a_section_split_across_chunks() -> None:
    merged = merge(
        [
            _part([("Intro", ["a"]), ("Raft", ["b", "c"])], [300, 40], "First half."),
            _part([("raft", ["c", "d"]), ("Paxos", ["e"])], [500], "Second half."),
        ]
    )
    assert [(s.heading, s.bullets) for s in merged.sections] == [
        ("Intro", ["a"]),
        ("Raft", ["b", "c", "d"]),
        ("Paxos", ["e"]),
    ]
    assert [f.t_sec for f in merged.frames] == [40, 300, 500]
    assert merged.title == "Notes First half."
    assert merged.summary == "First half. Second half."


def test_merge_keeps_section_starts_in_order_and_asks_each_question_once() -> None:
    merged = merge(
        [
            _part([("Intro", ["a"]), ("Raft", ["b"])], [], "1", starts=[0, 300],
                  questions=["Why a leader?", "What is a term?"]),
            # The parts overlap at the seam: a new topic may claim a start the first part covers.
            _part([("raft", ["c"]), ("Paxos", ["e"]), ("Zab", ["f"])], [], "2",
                  starts=[560, 280, 700], questions=["what is a term?", "Why Paxos?"]),
        ]
    )  # fmt: skip
    assert [(s.heading, s.start_sec) for s in merged.sections] == [
        ("Intro", 0),
        ("Raft", 300),  # the continued section keeps its first start
        ("Paxos", 300),  # never before the section above it
        ("Zab", 700),
    ]
    assert [q.question for q in merged.questions] == [
        "Why a leader?",
        "What is a term?",
        "Why Paxos?",
    ]
