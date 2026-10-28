"""SQLAlchemy tables (jobs, stage_checkpoints, job_events) and session factory (ARCHITECTURE §5).

The Notion token is never a column here (F-14, invariant 2).
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    Table,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ENUM, JSONB, UUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from common.state import JobStatus

metadata = MetaData(
    naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_name)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

job_status_enum = ENUM(
    *[status.value for status in JobStatus], name="job_status", create_type=False
)

jobs = Table(
    "jobs",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("video_id", Text, nullable=False),
    Column("youtube_url", Text, nullable=False),
    Column("status", job_status_enum, nullable=False, server_default=JobStatus.QUEUED.value),
    Column("progress", SmallInteger, nullable=False, server_default=text("0")),
    Column("video_title", Text),
    Column("duration_sec", Integer),
    Column("notion_parent_id", Text),
    Column("notion_page_id", Text),
    Column("notion_page_url", Text),
    Column("frames_requested", SmallInteger),
    Column("frames_published", SmallInteger),
    Column("warnings", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("error", JSONB),
    Column("waiting_until", DateTime(timezone=True)),
    Column("waiting_reason", Text),
    Column("idempotency_key", Text, unique=True),
    Column("enqueued_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("finished_at", DateTime(timezone=True)),
    CheckConstraint("progress BETWEEN 0 AND 100", name="progress_range"),
    # Outbox: rows whose first message was never confirmed (ARCHITECTURE §10).
    Index("ix_jobs_outbox", "created_at", postgresql_where=text("enqueued_at IS NULL")),
)

stage_checkpoints = Table(
    "stage_checkpoints",
    metadata,
    Column(
        "job_id",
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("stage", Text, primary_key=True),
    Column("completed_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("output", JSONB, nullable=False),
)

job_events = Table(
    "job_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("job_id", UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE")),
    Column("at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("kind", Text, nullable=False),
    Column("from_status", job_status_enum),
    Column("to_status", job_status_enum),
    Column("stage", Text),
    Column("attempt", SmallInteger),
    Column("detail", JSONB),
    Index("ix_job_events_job_id_at", "job_id", "at"),
)


def create_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
