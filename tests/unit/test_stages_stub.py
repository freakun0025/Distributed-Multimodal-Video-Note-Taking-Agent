"""M1 stub stages, the atomic artefact writer and StageOutcome guards."""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from common.config import Settings
from common.logging import get_logger
from common.models import StageMessage
from common.state import JobStatus, Stage
from worker.artifacts import write_json, write_json_atomic
from worker.context import JobRecord, StageContext, StageOutcome
from worker.stages import STAGES


def test_every_stage_has_an_implementation() -> None:
    assert set(STAGES) == set(Stage)


@pytest.mark.parametrize("stage", list(Stage))
async def test_stub_stage_reports_progress_and_writes_its_artefact(
    tmp_path: Path, stage: Stage
) -> None:
    job_id = uuid4()
    progress: list[float] = []

    async def report(fraction: float) -> None:
        progress.append(fraction)

    ctx = StageContext(
        job=JobRecord(
            id=job_id,
            video_id="dQw4w9WgXcQ",
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            status=JobStatus.QUEUED,
            progress=0,
            video_title=None,
            duration_sec=None,
            notion_parent_id=None,
            notion_page_id=None,
            notion_page_url=None,
            warnings=(),
            enqueued_at=None,
        ),
        message=StageMessage(job_id=job_id, stage=stage, attempt=2, notion_token_enc="enc"),
        job_dir=tmp_path / str(job_id),
        settings=Settings(_env_file=None, stub_stage_seconds=0),  # type: ignore[call-arg]
        log=get_logger(),
        report_progress=report,
    )
    outcome = await STAGES[stage](ctx)
    assert progress == [0.25, 0.5, 0.75, 1.0]
    artefact = ctx.job_dir / outcome.summary["artefact"]
    assert json.loads(artefact.read_text(encoding="utf-8")) == {
        "stage": stage.value,
        "attempt": 2,
    }


def test_atomic_write_replaces_the_file_and_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "jobs" / "x" / "meta.json"
    write_json_atomic(target, {"v": 1})
    write_json_atomic(target, {"v": 2, "title": "Grüße"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2, "title": "Grüße"}
    assert [p.name for p in target.parent.iterdir()] == ["meta.json"]


def test_a_failed_write_keeps_the_previous_file(tmp_path: Path) -> None:
    target = tmp_path / "synthesis.json"
    write_json_atomic(target, {"v": 1})
    with pytest.raises(TypeError):
        write_json_atomic(target, {"v": object()})
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["synthesis.json"]


async def test_async_write_runs_off_the_event_loop(tmp_path: Path) -> None:
    await write_json(tmp_path / "a.json", [1, 2])
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8")) == [1, 2]


def test_stages_cannot_set_the_job_status_through_their_outcome() -> None:
    with pytest.raises(ValueError):
        StageOutcome(summary={}, job_updates={"status": "COMPLETED"})
    assert StageOutcome(summary={}, job_updates={"video_title": "t"}).job_updates
