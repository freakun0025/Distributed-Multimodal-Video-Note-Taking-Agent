"""Queue resilience (Spec §9): a burst of jobs, each its own job, drained with zero LLM 429s.

    make burst [jobs=6]           fakes stack: one job per video, back to back, then the report
    make burst-report [since=15]  the report for every job submitted in the last `since` minutes,
                                  e.g. after clicking Take Notes on several videos (real stack)

The report lists each job's status and how long it took, when the burst drained, and every
answer the LLM gave these jobs, by HTTP status. The answers are read from the workers' logs (the
HTTP client logs each Gemini call with its status, on the real stack as on the fakes), so a 429
from Gemini is counted even though the workers retry it. Exit 1 when a job did not complete or
the LLM answered 429.

`submit` needs a Notion token for each job: on the fakes any string works; the real stack only
gets jobs from the extension (the token lives there), hence `report`. Runs on the host like
chaos/kill_worker.py: the gateway on localhost, Postgres on the port Compose publishes
(`CHAOS_DATABASE_URL`), and the docker CLI for the logs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from common.db import create_engine, jobs
from common.state import JobStatus, is_terminal

GATEWAY_URL = os.environ.get("CHAOS_GATEWAY_URL", "http://localhost:8000")
FAKES_URL = os.environ.get("CHAOS_FAKES_URL", "http://127.0.0.1:8090")
DATABASE_URL = os.environ.get(
    "CHAOS_DATABASE_URL", "postgresql+asyncpg://vna:vna@localhost:5433/vna"
)
FAKE_NOTION_TOKEN = "burst-fake-notion-token"  # the fake Notion accepts any token
POLL_S = 2.0
SUCCESS = frozenset({JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS})
# httpx's log line for each LLM call, e.g. `HTTP Request: POST https://generativelanguage.
# googleapis.com/v1beta/models/gemini-3.6-flash:generateContent "HTTP/1.1 429 Too Many Requests"`.
_LLM_CALL = re.compile(r'POST \S*/models/([^/:\s]+):generateContent "HTTP/[\d.]+ (\d{3})')


@dataclass(frozen=True)
class JobRow:
    id: UUID
    video_id: str
    title: str | None
    status: JobStatus
    created_at: datetime
    finished_at: datetime | None
    error_code: str | None


@dataclass(frozen=True)
class LlmAnswer:
    job_id: str
    model: str
    status: int


def video_ids(count: int) -> list[str]:
    """`count` different, valid YouTube-style IDs (11 characters), new on every run."""
    return [f"burst{secrets.token_hex(3)}" for _ in range(count)]


async def submit(gateway: httpx.AsyncClient, ids: list[str], token: str) -> list[UUID]:
    """One POST /jobs per video, back to back, as several Take Notes clicks would send."""
    submitted = []
    for video_id in ids:
        body = {"youtube_url": f"https://www.youtube.com/watch?v={video_id}", "notion_token": token}
        response = await gateway.post("/jobs", json=body)
        response.raise_for_status()
        submitted.append(UUID(response.json()["job_id"]))
    return submitted


async def load(engine: AsyncEngine, job_ids: list[UUID]) -> list[JobRow]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            select(
                jobs.c.id,
                jobs.c.video_id,
                jobs.c.video_title,
                jobs.c.status,
                jobs.c.created_at,
                jobs.c.finished_at,
                jobs.c.error,
            )
            .where(jobs.c.id.in_(job_ids))
            .order_by(jobs.c.created_at)
        )
        return [
            JobRow(
                id=row.id,
                video_id=row.video_id,
                title=row.video_title,
                status=JobStatus(row.status),
                created_at=row.created_at,
                finished_at=row.finished_at,
                error_code=(row.error or {}).get("code"),
            )
            for row in rows
        ]


async def jobs_since(engine: AsyncEngine, minutes: float) -> list[UUID]:
    async with engine.connect() as conn:
        rows = await conn.scalars(
            select(jobs.c.id)
            .where(jobs.c.created_at >= func.now() - timedelta(minutes=minutes))
            .order_by(jobs.c.created_at)
        )
        return list(rows.all())


async def wait_drained(
    engine: AsyncEngine, job_ids: list[UUID], timeout_s: float, *, quiet: bool = False
) -> list[JobRow]:
    """Poll until every job is terminal (or the timeout); print each one as it finishes."""
    deadline = time.monotonic() + timeout_s
    seen: set[UUID] = set()
    while True:
        rows = await load(engine, job_ids)
        for row in rows:
            if is_terminal(row.status) and row.id not in seen:
                seen.add(row.id)
                if not quiet:
                    _say(f"  {str(row.id)[:8]} {row.status.value}")
        if all(is_terminal(row.status) for row in rows) or time.monotonic() >= deadline:
            return rows
        await asyncio.sleep(POLL_S)


def parse_llm_answers(log_lines: list[str], job_ids: set[str]) -> list[LlmAnswer]:
    """The LLM's answers to these jobs, from the workers' JSON log lines."""
    answers = []
    for line in log_lines:
        start = line.find("{")
        if start < 0:
            continue
        try:
            entry: dict[str, Any] = json.loads(line[start:])
        except ValueError:
            continue
        match = _LLM_CALL.search(str(entry.get("event", "")))
        if match and entry.get("job_id") in job_ids:
            answers.append(LlmAnswer(str(entry["job_id"]), match[1], int(match[2])))
    return answers


async def worker_logs(since: datetime) -> list[str]:
    """Every worker's log lines since `since` (docker compose, run from the repository)."""
    command = ["docker", "compose", "logs", "--no-log-prefix", "--since", since.isoformat()]
    done = await asyncio.to_thread(
        subprocess.run, [*command, "worker"], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f"docker compose logs failed: {done.stderr.strip()[-200:]}")
    return done.stdout.splitlines()


def _clock(delta: timedelta) -> str:
    seconds = max(0, round(delta.total_seconds()))
    return f"{seconds // 60}:{seconds % 60:02d}"


def report(rows: list[JobRow], answers: list[LlmAnswer]) -> bool:
    """Print the burst's outcome; True when every job succeeded and the LLM never said 429."""
    if not rows:
        _say("No jobs found.")
        return False
    first = rows[0].created_at
    _say(f"\n{'job':<9} {'video':<12} {'status':<24} finished after  title")
    for row in rows:
        took = _clock(row.finished_at - row.created_at) if row.finished_at else "-"
        status = row.status.value + (f" ({row.error_code})" if row.error_code else "")
        title = (row.title or "")[:40]
        _say(f"{str(row.id)[:8]:<9} {row.video_id:<12} {status:<24} {took:>14}  {title}")
    finished = [row.finished_at for row in rows if row.finished_at]
    all_done = all(is_terminal(row.status) for row in rows)
    succeeded = sum(row.status in SUCCESS for row in rows)
    if all_done and finished:
        drained = f"drained {_clock(max(finished) - first)} after the first job was submitted"
    else:
        drained = "not drained yet"
    _say(f"\n{len(rows)} jobs, {succeeded} completed; {drained}.")
    by_status = Counter(answer.status for answer in answers)
    by_model = Counter(answer.model for answer in answers)
    statuses = ", ".join(f"{count} x {status}" for status, count in sorted(by_status.items()))
    models = ", ".join(f"{model}: {count}" for model, count in by_model.most_common())
    _say(f"LLM answers to these jobs: {statuses or 'none'} ({models or '-'})")
    too_many = by_status.get(429, 0)
    _say(f"429s from the LLM: {too_many}")
    return all_done and succeeded == len(rows) and too_many == 0


def _say(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


async def _is_fakes_stack() -> bool:
    try:
        async with httpx.AsyncClient(base_url=FAKES_URL, timeout=2) as fakes:
            return (await fakes.get("/healthz")).status_code == 200
    except httpx.HTTPError:
        return False


async def main(args: argparse.Namespace) -> int:
    engine = create_engine(args.database_url)
    try:
        if args.command == "submit":
            token = os.environ.get("BURST_NOTION_TOKEN")
            if token is None:
                if not await _is_fakes_stack():
                    _say(
                        "The real stack takes jobs from the extension: click Take Notes on "
                        "several videos in a row, then run `make burst-report`."
                    )
                    return 2
                token = FAKE_NOTION_TOKEN
            async with httpx.AsyncClient(base_url=args.gateway, timeout=10) as gateway:
                job_ids = await submit(gateway, video_ids(args.jobs), token)
            _say(f"Submitted {len(job_ids)} jobs, one per video; waiting for them to finish ...")
        else:
            job_ids = await jobs_since(engine, args.since)
            _say(f"{len(job_ids)} jobs submitted in the last {args.since:g} minutes")
            if job_ids:
                _say("Waiting for every one to finish ...")
        if not job_ids:
            return 1
        rows = await wait_drained(engine, job_ids, args.timeout)
        logs = await worker_logs(rows[0].created_at - timedelta(seconds=5))
        answers = parse_llm_answers(logs, {str(job_id) for job_id in job_ids})
        return 0 if report(rows, answers) else 1
    finally:
        await engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gateway", default=GATEWAY_URL)
    parser.add_argument("--database-url", default=DATABASE_URL)
    parser.add_argument("--timeout", type=float, default=900, help="seconds to wait for a drain")
    commands = parser.add_subparsers(dest="command", required=True)
    submit_cmd = commands.add_parser("submit", help="submit a burst (fakes stack), then report")
    submit_cmd.add_argument("--jobs", type=int, default=6)
    report_cmd = commands.add_parser("report", help="report the jobs of the last minutes")
    report_cmd.add_argument("--since", type=float, default=15, help="minutes")
    return parser


if __name__ == "__main__":
    sys.exit(asyncio.run(main(_parser().parse_args())))
