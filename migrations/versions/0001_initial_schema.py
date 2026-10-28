"""initial schema: jobs, stage_checkpoints, job_events (ARCHITECTURE §5)

Revision ID: 0001
Revises:
Create Date: 2026-10-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JOB_STATUSES = (
    "QUEUED",
    "TRANSCRIBING",
    "SUMMARIZING",
    "EXTRACTING_FRAMES",
    "PUBLISHING",
    "COMPLETED",
    "COMPLETED_WITH_WARNINGS",
    "FAILED",
    "DEAD_LETTERED",
)
job_status = postgresql.ENUM(*JOB_STATUSES, name="job_status", create_type=False)


def upgrade() -> None:
    postgresql.ENUM(*JOB_STATUSES, name="job_status").create(op.get_bind())

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("video_id", sa.Text(), nullable=False),
        sa.Column("youtube_url", sa.Text(), nullable=False),
        sa.Column("status", job_status, server_default="QUEUED", nullable=False),
        sa.Column("progress", sa.SmallInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("video_title", sa.Text(), nullable=True),
        sa.Column("duration_sec", sa.Integer(), nullable=True),
        sa.Column("notion_parent_id", sa.Text(), nullable=True),
        sa.Column("notion_page_id", sa.Text(), nullable=True),
        sa.Column("notion_page_url", sa.Text(), nullable=True),
        sa.Column("frames_requested", sa.SmallInteger(), nullable=True),
        sa.Column("frames_published", sa.SmallInteger(), nullable=True),
        sa.Column(
            "warnings",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("error", postgresql.JSONB(), nullable=True),
        sa.Column("waiting_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("waiting_reason", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.Text(), nullable=True),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("progress BETWEEN 0 AND 100", name="ck_jobs_progress_range"),
        sa.PrimaryKeyConstraint("id", name="pk_jobs"),
        sa.UniqueConstraint("idempotency_key", name="uq_jobs_idempotency_key"),
    )
    op.create_index(
        "ix_jobs_outbox",
        "jobs",
        ["created_at"],
        postgresql_where=sa.text("enqueued_at IS NULL"),
    )

    op.create_table(
        "stage_checkpoints",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("output", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name="fk_stage_checkpoints_job_id_jobs", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("job_id", "stage", name="pk_stage_checkpoints"),
    )

    op.create_table(
        "job_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("from_status", job_status, nullable=True),
        sa.Column("to_status", job_status, nullable=True),
        sa.Column("stage", sa.Text(), nullable=True),
        sa.Column("attempt", sa.SmallInteger(), nullable=True),
        sa.Column("detail", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name="fk_job_events_job_id_jobs", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_job_events"),
    )
    op.create_index("ix_job_events_job_id_at", "job_events", ["job_id", "at"])


def downgrade() -> None:
    op.drop_index("ix_job_events_job_id_at", table_name="job_events")
    op.drop_table("job_events")
    op.drop_table("stage_checkpoints")
    op.drop_index("ix_jobs_outbox", table_name="jobs")
    op.drop_table("jobs")
    postgresql.ENUM(name="job_status").drop(op.get_bind())
