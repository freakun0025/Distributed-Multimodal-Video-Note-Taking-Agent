from common.db import job_events, job_status_enum, jobs, metadata, stage_checkpoints
from common.state import JobStatus


def test_schema_has_the_three_documented_tables() -> None:
    assert set(metadata.tables) == {"jobs", "stage_checkpoints", "job_events"}


def test_job_status_enum_matches_the_state_machine() -> None:
    assert list(job_status_enum.enums) == [status.value for status in JobStatus]


def test_no_column_can_hold_the_notion_token() -> None:
    for table in (jobs, stage_checkpoints, job_events):
        assert not [column for column in table.columns if "token" in column.name]


def test_checkpoints_are_unique_per_job_and_stage() -> None:
    assert [column.name for column in stage_checkpoints.primary_key] == ["job_id", "stage"]


def test_outbox_index_covers_unpublished_jobs_only() -> None:
    (index,) = [index for index in jobs.indexes if index.name == "ix_jobs_outbox"]
    assert str(index.dialect_options["postgresql"]["where"]) == "enqueued_at IS NULL"
