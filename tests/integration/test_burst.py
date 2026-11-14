"""Queue resilience (Spec §9) on the fakes stack: a burst of jobs drains with zero LLM 429s.

The fake LLM answers 429 to a model's call past FAKE_LLM_RPM in the last minute, like Gemini's
free tier, so the burst is measured against a real limit: only the workers' shared limiter
(LLM_RPM, one Redis bucket for every worker) keeps the count at zero. Same code as `make burst`
(chaos/burst.py). Needs `make up-fakes`.
"""

import shutil
from collections import Counter
from collections.abc import AsyncIterator
from datetime import timedelta

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from chaos.burst import (
    FAKE_NOTION_TOKEN,
    parse_llm_answers,
    submit,
    video_ids,
    wait_drained,
    worker_logs,
)
from common.state import JobStatus
from fakes.app import FAKE_LLM_RPM
from tests.integration.conftest import FAKES_URL, JobRows
from tests.integration.test_pipeline import _notion_pages

pytestmark = pytest.mark.integration

BURST = 6  # "a half-dozen submitted videos" (Spec §9)
# Six LLM calls at LLM_RPM 4 take over a minute; jobs the limiter turns away wait in delay
# queues (10 s, 30 s, 90 s) for their turn.
DRAIN_TIMEOUT_S = 600


@pytest.fixture
async def fakes() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=FAKES_URL, timeout=10) as client:
        await client.post("/_control/reset")  # the default limit, an empty minute
        yield client
        await client.post("/_control/reset")


async def test_a_burst_of_jobs_for_different_videos_drains_with_zero_llm_429s(
    gateway: httpx.AsyncClient, engine: AsyncEngine, job_rows: JobRows, fakes: httpx.AsyncClient
) -> None:
    switches = (await fakes.get("/_control/switches")).json()
    assert switches["llm_rpm"] == FAKE_LLM_RPM > 0  # a real limit to stay under

    ids = video_ids(BURST)
    job_ids = await submit(gateway, ids, FAKE_NOTION_TOKEN)  # back to back, like clicks
    for job_id in job_ids:
        job_rows.track(str(job_id))
    rows = await wait_drained(engine, job_ids, DRAIN_TIMEOUT_S, quiet=True)

    assert [row.status for row in rows] == [JobStatus.COMPLETED] * BURST, rows
    assert sorted(row.video_id for row in rows) == sorted(ids)  # each its own job
    llm = Counter(call["status"] for call in (await fakes.get("/_control/llm/calls")).json())
    assert llm[429] == 0, llm
    assert llm[200] >= BURST  # at least one synthesis call per job
    for job_id in job_ids:
        assert len(await _notion_pages(job_id)) == 1
    if shutil.which("docker") is not None:
        # The report's count, from the workers' logs, sees the same answers.
        logs = await worker_logs(rows[0].created_at - timedelta(seconds=5))
        answers = parse_llm_answers(logs, {str(job_id) for job_id in job_ids})
        assert len(answers) >= BURST
        assert {answer.status for answer in answers} == {200}
