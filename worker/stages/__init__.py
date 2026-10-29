"""Stage registry: one coroutine per pipeline stage (ARCHITECTURE §6.3, §8)."""

from __future__ import annotations

from common.state import Stage
from worker.context import StageFn
from worker.stages import frames, publish, summarize, transcribe

STAGES: dict[Stage, StageFn] = {
    Stage.TRANSCRIBE: transcribe.run,
    Stage.SUMMARIZE: summarize.run,
    Stage.EXTRACT_FRAMES: frames.run,
    Stage.PUBLISH: publish.run,
}
