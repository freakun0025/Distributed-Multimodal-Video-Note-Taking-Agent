"""`make kill-worker`: the killed container comes back once another worker has the message."""

from __future__ import annotations

import argparse
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from chaos import kill_worker
from chaos.kill_worker import Recovery, Running
from common.state import JobStatus, Stage

KILLED_AT = datetime(2026, 11, 14, 8, 48, 11, tzinfo=UTC)


class _Engine:
    async def dispose(self) -> None:
        pass


async def _run(
    monkeypatch: pytest.MonkeyPatch, *, redelivered: bool, leave_dead: bool = False
) -> list[str]:
    """Run main() with the database and docker replaced; return what happened, in order."""
    events: list[str] = []
    running = Running(uuid4(), "b3b8cb833e00")

    async def wait_running(*args: Any) -> Running:
        return running

    async def kill(engine: Any, worker: str) -> datetime:
        events.append(f"kill {worker}")
        return KILLED_AT

    async def docker(*args: str) -> None:
        events.append("docker " + " ".join(args))

    async def follow(
        engine: Any,
        job_id: Any,
        stage: Stage,
        killed_at: datetime,
        timeout_s: float,
        on_redelivered: Callable[[datetime, str], Awaitable[None]] | None = None,
    ) -> Recovery:
        at = killed_at + timedelta(seconds=3)
        if redelivered:
            events.append("redelivered")
            assert on_redelivered is not None
            await on_redelivered(at, "f9139d0b79a3")
        events.append("job finished")
        return Recovery(
            killed_at=killed_at,
            redelivered_at=at if redelivered else None,
            redelivered_to="f9139d0b79a3" if redelivered else None,
            finished_at=killed_at + timedelta(seconds=40),
            status=JobStatus.COMPLETED,
        )

    monkeypatch.setattr(kill_worker, "create_engine", lambda url: _Engine())
    monkeypatch.setattr(kill_worker, "wait_running", wait_running)
    monkeypatch.setattr(kill_worker, "kill", kill)
    monkeypatch.setattr(kill_worker, "docker", docker)
    monkeypatch.setattr(kill_worker, "follow", follow)
    args = argparse.Namespace(
        stage="PUBLISH",
        job=None,
        wait=1.0,
        timeout=1.0,
        leave_dead=leave_dead,
        database_url="postgresql+asyncpg://unused",
    )
    assert await kill_worker.main(args) == 0
    return events


async def test_the_killed_worker_is_started_again_once_another_worker_has_the_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events = await _run(monkeypatch, redelivered=True)
    # Back before the job finishes: the rest of the job does not run on one worker.
    assert events == [
        "kill b3b8cb833e00",
        "redelivered",
        "docker start b3b8cb833e00",
        "job finished",
    ]
    out = capsys.readouterr().out
    assert "Redelivered to worker f9139d0b79a3 3.0 s after the kill" in out
    assert out.index("Redelivered") < out.index("Started b3b8cb833e00 again")
    assert "Job COMPLETED 40.0 s after the kill" in out


async def test_without_a_redelivery_the_worker_is_started_again_at_the_end(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events = await _run(monkeypatch, redelivered=False)
    assert events == ["kill b3b8cb833e00", "job finished", "docker start b3b8cb833e00"]
    assert "No redelivery seen" in capsys.readouterr().out


async def test_leave_dead_keeps_the_killed_worker_down(monkeypatch: pytest.MonkeyPatch) -> None:
    events = await _run(monkeypatch, redelivered=True, leave_dead=True)
    assert not [event for event in events if event.startswith("docker")]
