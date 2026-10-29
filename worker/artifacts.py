"""Atomic artefact writes on the shared volume (ARCHITECTURE §5 artefacts)."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, data: Any) -> None:
    """Write `path` via a unique `*.tmp` sibling, fsync, then `os.replace`.

    Readers never see a half-written file, and two workers running the same stage (a duplicate
    delivery) never write into each other's temp file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


async def write_json(path: Path, data: Any) -> None:
    await asyncio.to_thread(write_json_atomic, path, data)
