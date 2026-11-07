"""ffmpeg helpers: grab one frame with its brightness and blur stats (ARCHITECTURE §8.3, D33).

Frames only ever come from the source video (C-B). Every grab is written to a temporary file;
the stage renames the one it keeps.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path

# signalstats: brightness on the video scale (black = 16); blurdetect: higher = blurrier, NaN
# when the frame has no edges at all.
_STATS_FILTER = "signalstats,blurdetect,metadata=mode=print:file=-"
_KEYS = {
    "lavfi.signalstats.YAVG": "luma",
    "lavfi.signalstats.YMIN": "luma_min",
    "lavfi.signalstats.YMAX": "luma_max",
    "lavfi.blur": "blur",
}
_STDERR_MAX_CHARS = 200


class FrameGrabError(Exception):
    """ffmpeg produced no frame (e.g. a time past the end, or an unreadable file)."""


@dataclass(frozen=True)
class Grab:
    at_s: float
    path: Path
    luma: float
    luma_min: float
    luma_max: float
    blur: float  # NaN: no edges (a flat frame)

    def blank_reason(self, *, blank_luma: float, flat_range: float) -> str | None:
        """Why this grab is unusable as a picture, or None."""
        if self.luma < blank_luma:
            return "blank frame (black)"
        if self.luma_max - self.luma_min < flat_range or math.isnan(self.blur):
            return "blank frame (flat)"
        return None


def parse_stats(output: str) -> dict[str, float]:
    """The first value of each metadata key; ffmpeg prints a second, unused frame after it."""
    stats: dict[str, float] = {}
    for line in output.splitlines():
        key, _, value = line.strip().partition("=")
        name = _KEYS.get(key)
        if name and name not in stats:
            stats[name] = float(value)  # "nan" / "-nan" parse to NaN
    return stats


async def grab_frame(
    video: Path, at_s: float, out: Path, *, quality: int, timeout_s: float
) -> Grab:
    """One JPEG from `video` at `at_s` seconds, written to `out`, with its stats."""
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{at_s:.3f}", "-i", str(video),
        "-frames:v", "1", "-vf", _STATS_FILTER, "-q:v", str(quality),
        "-f", "mjpeg", "-y", str(out),
    ]  # fmt: skip
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise FrameGrabError(f"ffmpeg timed out after {timeout_s:g}s") from None
    stats = parse_stats(stdout.decode("utf-8", "replace"))
    if process.returncode != 0 or not stats:  # with exit code 0 and stats, a frame was written
        detail = stderr.decode("utf-8", "replace").strip()[-_STDERR_MAX_CHARS:]
        raise FrameGrabError(detail or "ffmpeg produced no frame")
    return Grab(
        at_s=at_s,
        path=out,
        luma=stats.get("luma", math.nan),
        luma_min=stats.get("luma_min", math.nan),
        luma_max=stats.get("luma_max", math.nan),
        blur=stats.get("blur", math.nan),
    )


def sharpest(grabs: list[Grab]) -> Grab:
    """The least blurry grab; ties (and NaN) go to the earliest."""
    return min(grabs, key=lambda g: (math.isnan(g.blur), g.blur if not math.isnan(g.blur) else 0))
