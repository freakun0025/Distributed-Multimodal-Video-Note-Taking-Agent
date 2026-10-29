"""M1 stub stage: sleeps in steps, reports progress, writes one artefact (ROADMAP M1).

Each real stage replaces its stub in M2; this module goes when the last one does.
"""

from __future__ import annotations

import asyncio

from common.state import Stage
from worker.artifacts import write_json
from worker.context import StageContext, StageOutcome

_STEPS = 4


async def run_stub(ctx: StageContext, stage: Stage) -> StageOutcome:
    for step in range(1, _STEPS + 1):
        await asyncio.sleep(ctx.settings.stub_stage_seconds / _STEPS)
        await ctx.report_progress(step / _STEPS)
    artefact = ctx.job_dir / f"stub_{stage.value.lower()}.json"
    await write_json(artefact, {"stage": stage.value, "attempt": ctx.message.attempt})
    ctx.log.info("stub_stage_done", artefact=artefact.name)
    return StageOutcome(summary={"stub": True, "artefact": artefact.name})
