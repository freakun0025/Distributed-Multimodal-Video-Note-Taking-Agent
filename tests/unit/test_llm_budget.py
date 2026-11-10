"""Duration-scaled notes floor and frame minimum (CP7c, D35)."""

from typing import Any

import pytest

from common.config import Settings
from worker.llm.budget import FrameSlot, NotesFloor, frame_slots, min_frames_for, notes_floor


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


DEFAULTS = _settings()  # a section per 150 s, 2.5 bullets a minute, at least a frame per 180 s


@pytest.mark.parametrize(
    ("span_s", "floor"),
    [
        (1088, NotesFloor(sections=7, bullets=7)),  # the 18:08 StatQuest lecture: 49 bullets
        (1181, NotesFloor(sections=7, bullets=8)),  # the 19:41 history video: 56 bullets
        (1200, NotesFloor(sections=8, bullets=7)),  # MAX_VIDEO_SECONDS
        (648, NotesFloor(sections=4, bullets=7)),  # 10:48
        (120, NotesFloor(sections=1, bullets=5)),
        (300, NotesFloor(sections=2, bullets=7)),
        (600, NotesFloor(sections=4, bullets=7)),  # 25 bullets wanted, rounded up per section
        (30, NotesFloor(sections=1, bullets=2)),
        (0, NotesFloor(sections=1, bullets=1)),
    ],
)
def test_the_notes_floor_grows_with_the_length_of_the_part(span_s: int, floor: NotesFloor) -> None:
    assert notes_floor(span_s, DEFAULTS) == floor


def test_the_floor_is_capped_to_keep_the_schema_small() -> None:
    assert notes_floor(10 * 3600, DEFAULTS) == NotesFloor(sections=8, bullets=8)
    custom = _settings(notes_floor_max_sections=4, notes_floor_max_bullets=3)
    assert notes_floor(10 * 3600, custom) == NotesFloor(sections=4, bullets=3)


def test_no_bullets_per_minute_still_asks_for_one_bullet_per_section() -> None:
    assert notes_floor(1088, _settings(notes_bullets_per_minute=0)) == NotesFloor(7, 1)


@pytest.mark.parametrize(
    ("duration_s", "frames"),
    [(1088, 6), (1200, 6), (648, 3), (30, 3), (0, 3), (3000, 8)],
)
def test_minimum_frames_scale_with_duration_within_min_and_max(
    duration_s: int, frames: int
) -> None:
    assert min_frames_for(duration_s, DEFAULTS) == frames


def test_max_frames_wins_over_a_larger_min_frames() -> None:
    assert min_frames_for(60, _settings(min_frames=5, max_frames=4)) == 4


def test_frame_slots_split_the_part_into_equal_stretches_of_candidates() -> None:
    candidates = [15, 45, 89, 166, 438, 589, 740, 831, 1042, 1063]
    slots = frame_slots(0, 1087, 3, candidates)
    assert slots == [
        FrameSlot(0, 361, (15, 45, 89, 166)),
        FrameSlot(362, 724, (438, 589)),
        FrameSlot(725, 1087, (740, 831, 1042, 1063)),
    ]


def test_a_stretch_without_candidates_is_left_out() -> None:
    slots = frame_slots(0, 599, 3, [10, 20, 450])
    assert [s.choices for s in slots] == [(10, 20), (450,)]


def test_without_candidates_each_slot_is_a_range_of_seconds() -> None:
    assert frame_slots(100, 399, 3) == [
        FrameSlot(100, 199),
        FrameSlot(200, 299),
        FrameSlot(300, 399),
    ]
    assert frame_slots(0, 1, 5) == [FrameSlot(0, 0), FrameSlot(1, 1)]  # fewer seconds than slots
    assert frame_slots(0, 600, 0) == []
