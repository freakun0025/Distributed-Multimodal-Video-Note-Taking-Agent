"""PUBLISH (ARCHITECTURE §8.4, F-13): one page per job, frames uploaded, resume-safe append."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from common.errors import ErrorCode, PermanentError
from common.models import FrameResult, Synthesis
from common.state import Stage
from tests.unit.fakes import make_context
from worker.clients.notion import NotionPage
from worker.context import StageContext
from worker.notion_page import job_tag
from worker.stages.publish import publish

PARENT = "0123456789abcdef0123456789abcdef"
SYNTHESIS = {
    "title": "Raft",
    "summary": "How Raft agrees.",
    "sections": [
        {"heading": "Intro", "start_sec": 0, "bullets": ["a", "b"]},
        {"heading": "Leaders", "start_sec": 240, "bullets": ["c"]},
    ],
    "questions": [{"question": "Why a leader?", "answer": "One node orders writes."}],
    "frames": [
        {"t_sec": 20, "caption": "Title slide.", "reason": "Opens."},
        {"t_sec": 300, "caption": "State diagram.", "reason": "Shows the states."},
        {"t_sec": 500, "caption": "Lost frame.", "reason": "Failed."},
    ],
}


class MemoryNotion:
    """The NotionApi calls, in memory: pages under parents, appended blocks, uploads."""

    def __init__(self) -> None:
        self.children: dict[str, list[Any]] = {}
        self.titles: dict[str, str] = {}
        self.uploads: list[str] = []
        self.appends: list[int] = []
        self.fail_after_blocks: int | None = None  # simulate a crash part-way through

    async def child_pages(self, parent_id: str) -> list[tuple[str, str]]:
        return [(pid, self.titles[pid]) for pid in self.children.get(parent_id, [])
                if isinstance(pid, str) and pid in self.titles]  # fmt: skip

    async def count_children(self, block_id: str) -> int:
        return len(self.children.get(block_id, []))

    async def page(self, page_id: str) -> NotionPage:
        return NotionPage(id=page_id, url=f"https://notion.test/{page_id}")

    async def create_page(self, parent_id: str, title: str) -> NotionPage:
        page_id = f"page{len(self.titles) + 1}"
        self.titles[page_id] = title
        self.children.setdefault(parent_id, []).append(page_id)
        return await self.page(page_id)

    async def append(self, block_id: str, children: Sequence[Mapping[str, Any]]) -> None:
        on_page = self.children.setdefault(block_id, [])
        for child in children:
            if self.fail_after_blocks is not None and len(on_page) >= self.fail_after_blocks:
                raise RuntimeError("worker killed mid-append")
            on_page.append(dict(child))
        self.appends.append(len(children))

    async def upload_image(self, data: bytes, filename: str, content_type: str) -> str:
        assert content_type == "image/jpeg" and data == filename.encode()
        self.uploads.append(filename)
        return f"up-{filename}"


def _context(tmp_path: Path, **settings: Any) -> StageContext:
    settings.setdefault("notion_default_parent_id", PARENT)
    ctx = make_context(tmp_path, Stage.PUBLISH, **settings)
    job = ctx.job_dir
    (job / "frames").mkdir(parents=True)
    (job / "meta.json").write_text(
        json.dumps({"video_id": ctx.job.video_id, "title": "Raft lecture", "duration_sec": 600}),
        "utf-8",
    )
    (job / "synthesis.json").write_text(json.dumps(SYNTHESIS), "utf-8")
    manifest = [
        FrameResult(t_sec=20, path="frames/frame_00020.jpg"),
        FrameResult(t_sec=300, path="frames/frame_00300.jpg"),
        FrameResult(t_sec=500, error="blank frame"),  # a warning already; not published
    ]
    for frame in manifest:
        if frame.path:
            (job / frame.path).write_bytes(Path(frame.path).name.encode())
    (job / "frames" / "manifest.json").write_text(
        json.dumps([m.model_dump(mode="json", exclude_none=True) for m in manifest]), "utf-8"
    )
    (job / "video.mp4").write_bytes(b"video")
    (job / "preview.mp4").write_bytes(b"preview")  # SUMMARIZE's small copy
    (job / "video.f137.mp4.part").write_bytes(b"partial")  # a yt-dlp leftover
    return ctx


def _kinds(blocks: list[Any]) -> list[str]:
    return [b["type"] for b in blocks]


async def test_a_page_is_created_under_the_default_parent_and_filled(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    notion = MemoryNotion()
    outcome = await publish(ctx, notion)

    (page_id,) = notion.children[PARENT]
    assert notion.titles[page_id] == f"Raft lecture{job_tag(ctx.job.id)}"
    blocks = notion.children[page_id]
    assert _kinds(blocks) == [
        "video", "callout", "paragraph",
        "heading_2", "bulleted_list_item", "bulleted_list_item", "image",
        "heading_2", "bulleted_list_item", "image",
        "heading_2", "toggle",
    ]  # fmt: skip
    assert notion.uploads == ["frame_00020.jpg", "frame_00300.jpg"]  # the failed frame: no
    assert blocks[6]["image"]["file_upload"] == {"id": "up-frame_00020.jpg"}
    assert outcome.job_updates == {
        "notion_page_id": page_id,
        "notion_page_url": f"https://notion.test/{page_id}",
        "frames_published": 2,
    }
    # Both downloads go (finished or partial); the job's own files stay.
    assert sorted(p.name for p in ctx.job_dir.iterdir()) == [
        "frames", "meta.json", "synthesis.json",
    ]  # fmt: skip


async def test_a_retry_after_a_crash_finds_the_page_and_appends_only_the_rest(
    tmp_path: Path,
) -> None:
    ctx = _context(tmp_path)
    notion = MemoryNotion()
    notion.fail_after_blocks = 7  # the first frame (block 6) is on the page, the rest is not
    with pytest.raises(RuntimeError):
        await publish(ctx, notion)
    assert notion.uploads == ["frame_00020.jpg", "frame_00300.jpg"]

    notion.fail_after_blocks = None
    notion.uploads.clear()
    await publish(ctx, notion)
    (page_id,) = notion.children[PARENT]  # still one page
    assert len(notion.children[page_id]) == 12  # no block twice
    assert notion.uploads == ["frame_00300.jpg"]  # only frames not yet on the page


async def test_publishing_a_finished_page_again_changes_nothing(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    notion = MemoryNotion()
    await publish(ctx, notion)
    notion.appends.clear()
    notion.uploads.clear()
    await publish(ctx, notion)
    assert notion.appends == [] and notion.uploads == []


async def test_a_parent_page_link_from_the_extension_wins_over_the_default(
    tmp_path: Path,
) -> None:
    ctx = _context(tmp_path)
    other = "fedcba9876543210fedcba9876543210"
    message = ctx.message.model_copy(
        update={"notion_parent_id": f"https://www.notion.so/My-Notes-{other}"}
    )
    ctx = StageContext(**{**ctx.__dict__, "message": message})
    notion = MemoryNotion()
    await publish(ctx, notion)
    assert next(iter(notion.children)) == other and PARENT not in notion.children


@pytest.mark.parametrize(
    ("default", "given", "message"),
    [(None, None, "no destination page"), (PARENT, "not a page", "not a Notion page ID")],
)
async def test_without_a_usable_destination_the_job_fails_with_a_clear_reason(
    tmp_path: Path, default: str | None, given: str | None, message: str
) -> None:
    ctx = _context(tmp_path, notion_default_parent_id=default)
    if given:
        ctx = StageContext(
            **{
                **ctx.__dict__,
                "message": ctx.message.model_copy(update={"notion_parent_id": given}),
            }
        )
    with pytest.raises(PermanentError) as raised:
        await publish(ctx, MemoryNotion())
    assert raised.value.code is ErrorCode.NOTION_PARENT_NOT_FOUND
    assert message in raised.value.message


def test_the_synthesis_fixture_is_a_valid_contract() -> None:
    Synthesis.model_validate(SYNTHESIS)
