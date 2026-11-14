"""eval/: the constraint checks, the judge's request and verdict (O-5)."""

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from common.config import Settings
from common.models import FrameResult, Synthesis, Transcript, TranscriptSegment
from common.state import JobStatus
from eval.check_constraints import _real_frames, _status_agrees, check_job, dhash, distance
from eval.evidence import JobEvidence, eval_videos
from eval.judge import SCORES, Verdict, judge_request
from worker.llm.base import PromptImage

NOTES = {
    "title": "Raft",
    "summary": "How Raft reaches consensus.",
    "sections": [{"heading": "Leader election", "start_sec": 0, "bullets": ["Terms."]}],
    "questions": [{"question": "Why terms?", "answer": "To spot stale leaders."}],
    "frames": [
        {"t_sec": 30, "caption": "A state diagram.", "reason": "Shows the three states."},
        {"t_sec": 90, "caption": "A log table.", "reason": "Shows replication."},
    ],
}


def _job(tmp_path: Path, **overrides: object) -> JobEvidence:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(exist_ok=True)
    for t_sec in (30, 90):
        (frames_dir / f"frame_{t_sec:05d}.jpg").write_bytes(b"\xff\xd8\xff" + bytes([t_sec]))
    fields: dict[str, object] = {
        "job_id": uuid4(),
        "video_id": "dQw4w9WgXcQ",
        "title": "Raft explained",
        "status": JobStatus.COMPLETED,
        "duration_sec": 600,
        "frames_requested": 2,
        "frames_published": 2,
        "warnings": [],
        "created_at": datetime(2026, 11, 14, tzinfo=UTC),
        "folder": tmp_path,
        "synthesis": Synthesis.model_validate(NOTES),
        "transcript": Transcript(
            source="captions",
            segments=[TranscriptSegment(start=0, duration=30, text="Raft has terms.")],
        ),
        "frames": [
            FrameResult(t_sec=30, path="frames/frame_00030.jpg"),
            FrameResult(t_sec=90, path="frames/frame_00090.jpg"),
        ],
    }
    fields.update(overrides)
    return JobEvidence(**fields)  # type: ignore[arg-type]


def test_eval_videos_mark_the_held_out_one() -> None:
    videos = eval_videos()
    assert len(videos) >= 7
    assert [v.video_id for v in videos if v.held_out] == ["Gx5qb1uHss4"]


def test_status_agrees_when_every_frame_was_published(tmp_path: Path) -> None:
    check = _status_agrees(_job(tmp_path))
    assert check.ok is True
    assert "COMPLETED, 2/2 frames" in check.detail


def test_an_omitted_frame_must_be_warned_about_and_never_counted(tmp_path: Path) -> None:
    frames = [
        FrameResult(t_sec=30, path="frames/frame_00030.jpg"),
        FrameResult(t_sec=90, error="black frame"),
    ]
    silent = _job(tmp_path, frames=frames, frames_published=1)
    assert _status_agrees(silent).ok is False  # COMPLETED and no warning for the 90 s frame

    warned = _job(
        tmp_path,
        frames=frames,
        frames_published=1,
        status=JobStatus.COMPLETED_WITH_WARNINGS,
        warnings=[{"code": "FRAME_EXTRACTION_FAILED", "timestamp_sec": 90}],
    )
    assert _status_agrees(warned).ok is True


def test_a_captions_only_job_cannot_be_compared_with_candidates(tmp_path: Path) -> None:
    assert _real_frames(_job(tmp_path), {30: 0, 90: 0}).ok is None


def test_a_frame_far_from_its_candidate_fails_the_real_frame_check(tmp_path: Path) -> None:
    job = _job(tmp_path, candidates={30: tmp_path / "missing-30.jpg"})
    # The candidate can't be decoded: no hash to compare with.
    assert _real_frames(job, {30: 0}).ok is False


def test_hash_distance_counts_differing_bits() -> None:
    assert distance(0b1011, 0b0001) == 2
    assert distance(2**64 - 1, 0) == 64


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_dhash_matches_one_picture_at_two_sizes_and_tells_others_apart(tmp_path: Path) -> None:
    def picture(name: str, source: str, size: str) -> Path:
        out = tmp_path / name
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", f"{source}=size={size}",
             "-frames:v", "1", "-y", str(out)],
            check=True,
        )  # fmt: skip
        return out

    big = dhash(picture("big.jpg", "testsrc2", "1280x720"))
    small = dhash(picture("small.jpg", "testsrc2", "854x480"))
    other = dhash(picture("other.jpg", "mandelbrot", "1280x720"))
    assert distance(big, small) <= 12
    assert distance(big, other) > 12


def test_check_job_flags_a_frame_without_caption(tmp_path: Path) -> None:
    notes = {**NOTES, "frames": [NOTES["frames"][0]]}  # type: ignore[index]
    job = _job(tmp_path, synthesis=Synthesis.model_validate(notes))
    checks = {c.name: c for c in check_job(job, max_seconds=1200)}
    assert checks["B: every frame captioned"].ok is False
    assert "missing at [90]" in checks["B: every frame captioned"].detail
    assert checks["A: within the limit"].ok is True


def test_judge_sees_the_transcript_other_frames_notes_then_each_picture(tmp_path: Path) -> None:
    (tmp_path / "cand_00010.jpg").write_bytes(b"\xff\xd8\xffcand10")
    job = _job(
        tmp_path,
        candidates={10: tmp_path / "cand_00010.jpg", 30: tmp_path / "frames/frame_00030.jpg"},
    )
    request = judge_request(job)
    images = [part.data for part in request.parts if isinstance(part, PromptImage)]
    # The candidate not on the page, then the two pictures on the page, in order.
    assert images == [b"\xff\xd8\xffcand10", b"\xff\xd8\xff\x1e", b"\xff\xd8\xffZ"]
    prompt = request.prompt
    assert prompt.index("[0:00] Raft has terms.") < prompt.index("Frame at [0:10]")
    assert prompt.index("Frame at [0:10]") < prompt.index("## Leader election [0:00]")
    assert "Caption: A log table. Why: Shows replication." in prompt
    assert set(SCORES) <= set(request.schema["required"])


def test_a_verdict_passes_only_with_matching_captions_and_good_scores(tmp_path: Path) -> None:
    job = _job(tmp_path)
    good = {name: 5 for name in SCORES} | {
        "pictures": [{"t_sec": 30, "caption_matches_picture": True}]
    }
    assert Verdict(job, good).passed
    mismatch = good | {"pictures": [{"t_sec": 30, "caption_matches_picture": False}]}
    assert not Verdict(job, mismatch).passed
    assert not Verdict(job, good | {"coverage": 3}).passed
    assert not Verdict(job, None, "UPSTREAM_ERROR").passed


def test_the_judge_chain_is_parsed_and_checked_like_the_llm_chain() -> None:
    settings = Settings(_env_file=None, eval_judge_chain="gemini-x:low:30,gemini-y:medium:60")  # type: ignore[call-arg,arg-type]
    assert [spec.model for spec in settings.eval_judge_chain] == ["gemini-x", "gemini-y"]
    with pytest.raises(ValueError, match="EVAL_JUDGE_CHAIN lists a model twice"):
        Settings(_env_file=None, eval_judge_chain="gemini-x:low:30,gemini-x:low:30")  # type: ignore[call-arg,arg-type]


def test_by_default_the_judge_is_led_by_another_model_than_the_notes() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.eval_judge_chain[0].model != settings.llm_chain[0].model
