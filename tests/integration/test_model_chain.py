"""The LLM model chain end to end (D36): the fakes service makes the first model of LLM_CHAIN
answer 503 "high demand"; jobs still complete on the next model, and once the first model is
benched in Redis no worker calls it again.

Needs `make up-fakes`; reads LLM_CHAIN from the repo's .env like the workers do.
"""

import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from redis.asyncio import Redis

from common.config import Settings
from tests.integration.conftest import FAKES_URL, JobRows
from tests.integration.test_pipeline import _submit, _wait_terminal
from worker.llm.health import KEY_PREFIX

pytestmark = pytest.mark.integration

REDIS_URL = os.environ.get("ITEST_REDIS_URL", "redis://127.0.0.1:6380/0")


@pytest.fixture
async def chain() -> AsyncIterator[list[str]]:
    """The workers' chain, with clean model health before and after the test."""
    models = [spec.model for spec in Settings().llm_chain]
    if len(models) < 2:
        pytest.skip("LLM_CHAIN has a single model: nothing to step down to")
    redis = Redis.from_url(REDIS_URL)

    async def forget() -> None:
        keys = [f"{KEY_PREFIX}{kind}:{model}" for model in models for kind in ("bench", "strikes")]
        await redis.delete(*keys)

    await forget()
    yield models
    await forget()
    await redis.aclose()


@pytest.fixture
async def fakes() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=FAKES_URL, timeout=10) as client:
        yield client
        await client.put("/_control/llm", json={})  # every model answers again


async def _llm_calls(fakes: httpx.AsyncClient) -> list[tuple[str, int]]:
    calls: list[dict[str, Any]] = (await fakes.get("/_control/llm/calls")).json()
    return [(call["model"], call["status"]) for call in calls]


async def test_jobs_step_down_the_chain_when_a_model_keeps_failing(
    gateway: httpx.AsyncClient,
    job_rows: JobRows,
    fakes: httpx.AsyncClient,
    chain: list[str],
) -> None:
    first, second = chain[:2]
    redis = Redis.from_url(REDIS_URL)
    try:
        await fakes.put("/_control/llm", json={"unavailable": [first]})

        job = await _submit(gateway, job_rows)
        assert (await _wait_terminal(gateway, job))["status"] == "COMPLETED"
        # Attempt 1: one 503 is a blip, retried via the delay queue on the same model.
        # Attempt 2: the second 503 benches it, and the same attempt asks the next model.
        assert await _llm_calls(fakes) == [(first, 503), (first, 503), (second, 200)]
        assert await redis.pttl(f"{KEY_PREFIX}bench:{first}") > 0

        await fakes.put("/_control/llm", json={"unavailable": [first]})  # clears the call log
        job = await _submit(gateway, job_rows)
        assert (await _wait_terminal(gateway, job))["status"] == "COMPLETED"
        # Benched in Redis: the next job, on whichever worker, never calls it.
        assert await _llm_calls(fakes) == [(second, 200)]
    finally:
        await redis.aclose()
