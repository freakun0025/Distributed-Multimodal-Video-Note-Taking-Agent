"""Alembic environment: async migrations against DATABASE_URL (ARCHITECTURE §5)."""

from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy.engine import Connection

from common.config import get_settings
from common.db import create_engine, metadata
from common.logging import configure_logging

target_metadata = metadata


def _run(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def _run_online() -> None:
    engine = create_engine(get_settings().database_url)
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


def _run_offline() -> None:
    context.configure(
        url=get_settings().database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


configure_logging(get_settings().log_level, service="migrate")
if context.is_offline_mode():
    _run_offline()
else:
    asyncio.run(_run_online())
