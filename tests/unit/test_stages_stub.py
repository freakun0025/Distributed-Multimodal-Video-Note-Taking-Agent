"""The stage registry, the atomic artefact writer and StageOutcome guards."""

import json
from pathlib import Path

import pytest

from common.state import Stage
from worker.artifacts import write_json, write_json_atomic
from worker.context import StageOutcome
from worker.stages import STAGES


def test_every_stage_has_an_implementation() -> None:
    assert set(STAGES) == set(Stage)


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
