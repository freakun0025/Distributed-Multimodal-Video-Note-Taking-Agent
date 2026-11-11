"""The Notion page layout (ARCHITECTURE §8.4; user's CP8 decisions, D38)."""

from pathlib import Path
from uuid import UUID

from common.models import FramePick, Question, Section, Synthesis, VideoMeta
from worker.notion_page import (
    PENDING_UPLOAD,
    RICH_TEXT_MAX,
    PageFrame,
    build_page,
    page_title,
    place_frames,
    rich_text,
    with_upload,
)

WATCH = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
META = VideoMeta(video_id="dQw4w9WgXcQ", title="Raft", duration_sec=1067, channel="Prof")


def _synthesis(**overrides: object) -> Synthesis:
    data: dict[str, object] = {
        "title": "Raft",
        "summary": "How Raft agrees.",
        "sections": [
            Section(heading="Intro", start_sec=0, bullets=["a", "b"]),
            Section(heading="Leaders", start_sec=240, bullets=["c"]),
        ],
        "questions": [Question(question="Why a leader?", answer="One node orders writes.")],
        "frames": [],
    }
    data.update(overrides)
    return Synthesis.model_validate(data)


def _frame(t_sec: int) -> PageFrame:
    return PageFrame(t_sec, Path(f"frames/frame_{t_sec:05d}.jpg"), f"Slide {t_sec}", "Shows it")


def _text(block: dict[str, object]) -> str:
    body = block[block["type"]]  # type: ignore[index]
    return "".join(part["text"]["content"] for part in body["rich_text"])  # type: ignore[index]


def test_the_page_opens_with_the_video_then_the_link_and_the_summary() -> None:
    blocks = build_page(META, _synthesis(), [], watch_url=WATCH).blocks
    assert blocks[0] == {
        "object": "block",
        "type": "video",
        "video": {"type": "external", "external": {"url": WATCH}},
    }
    callout = blocks[1]["callout"]
    assert callout["rich_text"][0]["text"] == {
        "content": "Watch on YouTube",
        "link": {"url": WATCH},
    }
    assert _text(blocks[1]) == "Watch on YouTube · 17:47 · Prof"
    assert blocks[2]["type"] == "paragraph" and _text(blocks[2]) == "How Raft agrees."


def test_headings_link_to_their_moment_and_bullets_stay_flat() -> None:
    blocks = build_page(META, _synthesis(), [], watch_url=WATCH).blocks
    headings = [b for b in blocks if b["type"] == "heading_2"]
    assert [_text(h) for h in headings] == ["0:00 · Intro", "4:00 · Leaders", "Test yourself"]
    assert headings[1]["heading_2"]["rich_text"][0]["text"]["link"] == {"url": f"{WATCH}&t=240s"}
    bullets = [b for b in blocks if b["type"] == "bulleted_list_item"]
    assert [_text(b) for b in bullets] == ["a", "b", "c"]
    assert all("children" not in b["bulleted_list_item"] for b in bullets)


def test_each_frame_follows_the_section_whose_range_holds_it() -> None:
    sections = _synthesis().sections
    placed = place_frames(sections, [_frame(t) for t in (500, 5, 239, 240)])
    assert [[f.t_sec for f in group] for group in placed] == [[5, 239], [240, 500]]
    page = build_page(META, _synthesis(), [_frame(500), _frame(5)], watch_url=WATCH)
    kinds = [b["type"] for b in page.blocks]
    assert kinds == [
        "video", "callout", "paragraph",
        "heading_2", "bulleted_list_item", "bulleted_list_item", "image",
        "heading_2", "bulleted_list_item", "image",
        "heading_2", "toggle",
    ]  # fmt: skip
    assert {i: f.t_sec for i, f in page.images.items()} == {6: 5, 9: 500}
    image = page.blocks[6]["image"]
    assert image["file_upload"] == {"id": PENDING_UPLOAD}
    caption = image["caption"]
    assert caption[0]["text"]["content"] == "Slide 5 — why: Shows it "
    assert caption[1]["text"] == {"content": "(0:05)", "link": {"url": f"{WATCH}&t=5s"}}
    assert with_upload(page.blocks[6], "up-1")["image"]["file_upload"] == {"id": "up-1"}
    assert page.blocks[6]["image"]["file_upload"] == {"id": PENDING_UPLOAD}  # not mutated


def test_questions_are_toggles_with_the_answer_hidden_inside() -> None:
    toggle = build_page(META, _synthesis(), [], watch_url=WATCH).blocks[-1]
    assert toggle["type"] == "toggle" and _text(toggle) == "Why a leader?"
    (answer,) = toggle["toggle"]["children"]
    assert answer["type"] == "paragraph" and _text(answer) == "One node orders writes."


def test_without_questions_or_sections_nothing_is_left_dangling() -> None:
    page = build_page(META, _synthesis(sections=[], questions=[]), [_frame(30)], watch_url=WATCH)
    assert [b["type"] for b in page.blocks] == ["video", "callout", "paragraph", "image"]


def test_the_same_artefacts_always_give_the_same_blocks() -> None:
    frames = [_frame(5), _frame(300)]
    first = build_page(META, _synthesis(), frames, watch_url=WATCH)
    again = build_page(META, _synthesis(), list(reversed(frames)), watch_url=WATCH)
    assert first == again


def test_long_text_is_split_into_pieces_notion_accepts() -> None:
    pieces = rich_text("x" * (RICH_TEXT_MAX * 2 + 5))
    assert [len(p["text"]["content"]) for p in pieces] == [RICH_TEXT_MAX, RICH_TEXT_MAX, 5]
    assert rich_text("") == [{"type": "text", "text": {"content": "", "link": None}}]


def test_the_page_title_carries_the_job_tag() -> None:
    job = UUID("8f052215-f4bd-47e3-8eec-ca73de8b07dd")
    assert page_title(" Learn Claude Code ", job) == "Learn Claude Code · #8f052215"
    assert page_title("", job) == "Video notes · #8f052215"


def test_frame_picks_are_not_needed_to_build_the_page() -> None:
    # Captions come from PageFrame (manifest + picks joined by the stage), not synthesis.frames.
    synthesis = _synthesis(frames=[FramePick(t_sec=5, caption="c", reason="r")])
    assert build_page(META, synthesis, [], watch_url=WATCH).images == {}
