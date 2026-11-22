"""Automatic Constraint A/B checks on finished jobs, no LLM (Spec §7, §9; ARCHITECTURE §14, O-5).

    make check-constraints                  the newest successful job of each video in videos.json
    make check-constraints ids="<id> <id>"  these jobs

Constraint A, over every job in the database: no job longer than MAX_VIDEO_SECONDS succeeded or
reached the LLM (each one FAILED with VIDEO_TOO_LONG and has no synthesis.json), and every job
that succeeded is within the limit.

Constraint B, per job, on the frames its Notion page was built from:
  - caption: every published frame has an LLM-written caption and reason;
  - jpeg: every published frame is a JPEG ffmpeg can decode;
  - real frame: each frame is the same picture as the candidate frame the model saw at that
    second (both grabbed by ffmpeg from the video, at 720p and 480p): a 64-bit difference hash
    within DHASH_MATCH_BITS. For scale, the median distance to the job's other candidates is
    printed (another moment of the same video). A captions-only job has no candidates: "n/a";
  - status: frames_published, the status and the warnings agree with the frames manifest (a
    missing frame is omitted and reported, never substituted).

Exit 1 when any check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from common.config import get_settings
from common.db import create_engine, jobs
from common.state import JobStatus
from eval.evidence import (
    SUCCESS,
    JobEvidence,
    OutdatedJobError,
    eval_videos,
    latest_successful,
    load,
)

DHASH_MATCH_BITS = 12  # of 64: a resized, re-encoded copy of one picture stays well inside
FFMPEG_TIMEOUT_S = 30
JPEG_MAGIC = b"\xff\xd8\xff"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool | None  # None: not applicable
    detail: str


def dhash(image: Path) -> int:
    """64-bit difference hash: the picture shrunk to 9x8 grey pixels by ffmpeg, one bit per
    neighbouring pair (left brighter than right)."""
    done = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(image),
            "-vf", "scale=9:8:flags=area,format=gray", "-frames:v", "1",
            "-f", "rawvideo", "-",
        ],
        capture_output=True,
        timeout=FFMPEG_TIMEOUT_S,
        check=False,
    )  # fmt: skip
    pixels = done.stdout
    if done.returncode != 0 or len(pixels) != 72:
        raise ValueError(done.stderr.decode("utf-8", "replace").strip() or "not decodable")
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return bits


def distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def check_job(job: JobEvidence, max_seconds: int) -> list[Check]:
    checks = [
        Check(
            "A: within the limit",
            job.duration_sec is not None and job.duration_sec <= max_seconds,
            f"{job.duration_sec} s (limit {max_seconds} s)",
        )
    ]
    published = job.published

    missing = [f.t_sec for f in published if not _has_caption(job, f.t_sec)]
    checks.append(
        Check(
            "B: every frame captioned",
            not missing and job.synthesis is not None,
            f"{len(published) - len(missing)}/{len(published)} with caption and reason"
            + (f"; missing at {missing}" if missing else ""),
        )
    )

    hashes: dict[int, int] = {}
    broken: list[str] = []
    for frame in published:
        path = job.folder / str(frame.path)
        try:
            if path.read_bytes()[:3] != JPEG_MAGIC:
                raise ValueError("not a JPEG")
            hashes[frame.t_sec] = dhash(path)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            broken.append(f"{frame.t_sec}s: {exc}")
    checks.append(
        Check(
            "B: frames are JPEGs ffmpeg decodes",
            not broken,
            f"{len(hashes)}/{len(published)} decoded" + (f"; {broken}" if broken else ""),
        )
    )
    checks.append(_real_frames(job, hashes))
    checks.append(_status_agrees(job))
    return checks


def _has_caption(job: JobEvidence, t_sec: int) -> bool:
    found = job.caption(t_sec)
    return found is not None and all(text.strip() for text in found)


def _real_frames(job: JobEvidence, hashes: dict[int, int]) -> Check:
    name = "B: real frame (matches the candidate)"
    if not job.candidates:
        return Check(name, None, "n/a: captions-only job (no candidate frames to compare)")
    candidate_hashes: dict[int, int] = {}
    for t_sec, path in job.candidates.items():
        try:
            candidate_hashes[t_sec] = dhash(path)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
    own: list[int] = []
    unlike: list[int] = []
    bad: list[str] = []
    for t_sec, frame_hash in hashes.items():
        if t_sec not in candidate_hashes:
            bad.append(f"{t_sec}s: no candidate at that second")
            continue
        mine = distance(frame_hash, candidate_hashes[t_sec])
        own.append(mine)
        # Another scene, for scale: the median distance to the job's other candidates.
        others = sorted(distance(frame_hash, h) for t, h in candidate_hashes.items() if t != t_sec)
        if others:
            unlike.append(others[len(others) // 2])
        if mine > DHASH_MATCH_BITS:
            bad.append(f"{t_sec}s: {mine} bits from its candidate")
    detail = (
        f"{len(own)} frames, {min(own, default=0)}-{max(own, default=0)} bits from the candidate "
        f"at their second (limit {DHASH_MATCH_BITS}); other moments of the video: median "
        f"{sorted(unlike)[len(unlike) // 2] if unlike else 0} bits"
    )
    return Check(name, not bad and bool(own), detail + (f"; {bad}" if bad else ""))


def _status_agrees(job: JobEvidence) -> Check:
    ok_frames = job.published
    failed = sorted(f.t_sec for f in job.frames if not f.ok)
    warned = sorted(
        int(w["timestamp_sec"])
        for w in job.warnings
        if w.get("code") == "FRAME_EXTRACTION_FAILED" and w.get("timestamp_sec") is not None
    )
    expected = JobStatus.COMPLETED_WITH_WARNINGS if failed else JobStatus.COMPLETED
    problems = []
    if job.frames_published != len(ok_frames):
        problems.append(f"frames_published {job.frames_published} != {len(ok_frames)} on disk")
    if job.synthesis is not None and job.frames_requested != len(job.synthesis.frames):
        problems.append(f"frames_requested {job.frames_requested} != {len(job.synthesis.frames)}")
    if failed and job.status is not expected:
        problems.append(f"status {job.status.value} though frames {failed} failed")
    if failed and warned != failed:
        problems.append(f"warnings name {warned}, failed frames are {failed}")
    detail = f"{job.status.value}, {len(ok_frames)}/{len(job.frames)} frames" + (
        f", omitted {failed} with warnings" if failed else ""
    )
    return Check("B: status and warnings agree", not problems, "; ".join([detail, *problems]))


async def constraint_a_everywhere(engine: AsyncEngine, jobs_dir: Path, max_seconds: int) -> Check:
    """Every job in the database: none longer than the limit succeeded or reached the LLM."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(select(jobs.c.id, jobs.c.status, jobs.c.duration_sec, jobs.c.error))
        ).all()
    # A long video is rejected before its length is saved on the job: its error says so.
    rejected = [row for row in rows if (row.error or {}).get("code") == "VIDEO_TOO_LONG"]
    problems = []
    for row in rejected:
        if row.status != JobStatus.FAILED.value:
            problems.append(f"{str(row.id)[:8]} is {row.status} with VIDEO_TOO_LONG")
        if (jobs_dir / str(row.id) / "synthesis.json").exists():
            problems.append(f"{str(row.id)[:8]} reached the LLM")
    long_jobs = [r for r in rows if r.duration_sec is not None and r.duration_sec > max_seconds]
    problems += [f"{str(r.id)[:8]} ran though {r.duration_sec} s long" for r in long_jobs]
    # Jobs of the M1 skeleton ran stub stages (stub_transcribe.json, no real video): not evidence.
    stubs = {r.id for r in rows if (jobs_dir / str(r.id) / "stub_transcribe.json").exists()}
    succeeded = [row for row in rows if JobStatus(row.status) in SUCCESS and row.id not in stubs]
    unknown = [str(r.id)[:8] for r in succeeded if r.duration_sec is None]
    problems += [f"{job_id} succeeded without a known length" for job_id in unknown]
    detail = (
        f"{len(rejected)} jobs rejected as longer than {max_seconds} s, none reached the LLM; "
        f"{len(succeeded)} succeeded, all at most {max_seconds} s"
        + (f" ({len(stubs)} stub-pipeline jobs of M1 not counted)" if stubs else "")
    )
    return Check("A: over the limit -> rejected, never processed", not problems,
                 "; ".join(problems) if problems else detail)  # fmt: skip


def _mark(ok: bool | None) -> str:
    return {True: "PASS", False: "FAIL", None: "n/a "}[ok]


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    try:
        videos = {video.video_id: video for video in eval_videos()}
        if args.jobs:
            job_ids = [UUID(job) for job in args.jobs]
        else:
            newest = await latest_successful(engine, list(videos))
            job_ids = [newest[v] for v in videos if v in newest]
            for video_id in videos.keys() - newest.keys():
                print(f"(no successful job yet for {video_id})")
        all_ok = True
        overall = await constraint_a_everywhere(
            engine, settings.jobs_dir, settings.max_video_seconds
        )
        print(f"[{_mark(overall.ok)}] {overall.name}: {overall.detail}\n")
        all_ok &= overall.ok is not False
        for job_id in job_ids:
            try:
                job = await load(engine, settings.jobs_dir, job_id)
            except OutdatedJobError as exc:
                print(f"(skipped: {exc})\n")
                continue
            video = videos.get(job.video_id)
            label = " (held-out)" if video and video.held_out else ""
            print(f"{job.title or job.video_id}{label}")
            print(f"  job {job.job_id} of {job.created_at:%Y-%m-%d %H:%M} UTC, {job.video_id}")
            for check in await asyncio.to_thread(check_job, job, settings.max_video_seconds):
                all_ok &= check.ok is not False
                print(f"  [{_mark(check.ok)}] {check.name}: {check.detail}")
            print()
        print("All constraint checks passed." if all_ok else "Some constraint checks FAILED.")
        return 0 if all_ok else 1
    finally:
        await engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("jobs", nargs="*", help="job IDs (default: the videos in videos.json)")
    return parser


if __name__ == "__main__":
    sys.exit(asyncio.run(main(_parser().parse_args())))
