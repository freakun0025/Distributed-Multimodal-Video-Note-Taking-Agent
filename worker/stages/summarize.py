"""SUMMARIZE stage: chunked structured LLM synthesis, merged in order (ARCHITECTURE §8.2)."""

from __future__ import annotations

from common.state import Stage
from worker.context import StageContext, StageOutcome
from worker.stages.stub import run_stub


async def run(ctx: StageContext) -> StageOutcome:
    # M1 stub; the real stage (ARCHITECTURE §8.2) lands in M2.
    return await run_stub(ctx, Stage.SUMMARIZE)
