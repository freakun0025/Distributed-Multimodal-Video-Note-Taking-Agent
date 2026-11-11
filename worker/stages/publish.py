"""PUBLISH stage: find-or-create the Notion page, upload frames, resume-safe append (§8.4, F-13).

1. Find or create the page: the job's saved page, else a child page of the destination whose
   title ends with the job tag (listing the parent's children, never the search API), else a
   new empty page. A retried publish therefore never makes a second page.
2. Build the block list deterministically (worker/notion_page.py) and count the blocks already
   on the page (k); append blocks[k:] in batches, uploading each batch's frames just before the
   batch (Notion archives an upload not attached within the hour), so a crash mid-append resumes
   where it stopped instead of duplicating blocks.
3. Hand back the page ID and URL and the number of frames published; the runtime records them
   with the checkpoint and sets COMPLETED or COMPLETED_WITH_WARNINGS. Delete video.mp4.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from common.crypto import TokenCipher
from common.errors import ErrorCode, ExternalApi, PermanentError
from common.models import FrameResult, Synthesis, VideoMeta
from common.youtube_url import canonical_url
from worker.clients.notion import MAX_CHILDREN, NotionApi, NotionPage, notion_client, notion_id
from worker.context import StageContext, StageOutcome
from worker.notion_page import PageFrame, build_page, job_tag, page_title, with_upload
from worker.stages.frames import FRAMES_DIR, MANIFEST_FILE, VIDEO_NAME
from worker.stages.summarize import SYNTHESIS_FILE
from worker.stages.transcribe import META_FILE

FRAME_TYPE = "image/jpeg"  # ffmpeg writes JPEG frames (EXTRACT_FRAMES)


async def run(ctx: StageContext) -> StageOutcome:
    if ctx.settings.token_encryption_key is None:
        raise PermanentError(ErrorCode.TOKEN_DECRYPT_FAILED, "TOKEN_ENCRYPTION_KEY is not set")
    token = TokenCipher(ctx.settings.token_encryption_key).decrypt(ctx.message.notion_token_enc)
    async with notion_client(ctx.settings, ctx.limiter, token) as notion:
        return await publish(ctx, notion)


async def publish(ctx: StageContext, notion: NotionApi) -> StageOutcome:
    meta = VideoMeta.model_validate(await _read(ctx.job_dir / META_FILE))
    synthesis = Synthesis.model_validate(await _read(ctx.job_dir / SYNTHESIS_FILE))
    frames = _saved_frames(
        ctx.job_dir, synthesis, await _read(ctx.job_dir / FRAMES_DIR / MANIFEST_FILE)
    )

    page = await _find_or_create(ctx, notion, meta)
    await ctx.report_progress(0.1)
    built = build_page(meta, synthesis, frames, watch_url=canonical_url(ctx.job.video_id))
    present = await notion.count_children(page.id)
    if present > len(built.blocks):  # edited by hand since: never delete what is there
        ctx.log.warning("notion_page_longer_than_built", present=present, built=len(built.blocks))
    start = min(present, len(built.blocks))
    for batch_start in range(start, len(built.blocks), MAX_CHILDREN):
        indices = range(batch_start, min(batch_start + MAX_CHILDREN, len(built.blocks)))
        batch = []
        for i in indices:
            block = built.blocks[i]
            if (frame := built.images.get(i)) is not None:
                data = await asyncio.to_thread(frame.path.read_bytes)
                upload_id = await notion.upload_image(data, frame.path.name, FRAME_TYPE)
                block = with_upload(block, upload_id)
            batch.append(block)
        await notion.append(page.id, batch)
        await ctx.report_progress(0.1 + 0.9 * indices.stop / len(built.blocks))
    ctx.log.info(
        "notion_page_published",
        blocks=len(built.blocks),
        appended=len(built.blocks) - start,
        frames=len(frames),
        questions=len(synthesis.questions),
    )
    await asyncio.to_thread((ctx.job_dir / f"{VIDEO_NAME}.mp4").unlink, missing_ok=True)
    return StageOutcome(
        summary={
            "page_id": page.id,
            "blocks": len(built.blocks),
            "appended": len(built.blocks) - start,
        },
        job_updates={
            "notion_page_id": page.id,
            "notion_page_url": page.url,
            "frames_published": len(frames),
        },
    )


async def _find_or_create(ctx: StageContext, notion: NotionApi, meta: VideoMeta) -> NotionPage:
    if ctx.job.notion_page_id and ctx.job.notion_page_url:
        return NotionPage(id=ctx.job.notion_page_id, url=ctx.job.notion_page_url)
    raw = ctx.message.notion_parent_id or ctx.job.notion_parent_id
    parent = notion_id(raw or ctx.settings.notion_default_parent_id)
    if parent is None:
        raise PermanentError(
            ErrorCode.NOTION_PARENT_NOT_FOUND,
            "no destination page: set NOTION_DEFAULT_PARENT_ID or a parent page in the extension"
            if not raw
            else "the parent page ID is not a Notion page ID or link",
            api=ExternalApi.NOTION,
        )
    tag = job_tag(ctx.job.id)
    for page_id, title in await notion.child_pages(parent):
        if title.endswith(tag):
            ctx.log.info("notion_page_found")  # a previous attempt created it
            return await notion.page(page_id)
    page = await notion.create_page(parent, page_title(meta.title, ctx.job.id))
    ctx.log.info("notion_page_created")
    return page


def _saved_frames(job_dir: Path, synthesis: Synthesis, manifest: object) -> list[PageFrame]:
    """Frames EXTRACT_FRAMES saved, with the caption and reason the notes gave them; failed
    frames are already warnings (F-20) and are left out."""
    picks = {frame.t_sec: frame for frame in synthesis.frames}
    frames: list[PageFrame] = []
    for item in manifest if isinstance(manifest, list) else []:
        result = FrameResult.model_validate(item)
        pick = picks.get(result.t_sec)
        if result.path is not None and pick is not None:
            frames.append(PageFrame(result.t_sec, job_dir / result.path, pick.caption, pick.reason))
    return frames


async def _read(path: Path) -> object:
    return json.loads(await asyncio.to_thread(path.read_text, "utf-8"))
