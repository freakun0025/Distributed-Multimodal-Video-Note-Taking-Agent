"""Notion page IDs from what a user pastes; pure, shared by gateway and workers (§8.4)."""

from __future__ import annotations

import re

_HEX_RUN = re.compile(r"[0-9a-f]{32,}")


def notion_id(raw: str | None) -> str | None:
    """A page ID from what a user pastes: the bare ID, the dashed UUID or the page's URL
    (whose last 32 hex characters are the ID); None if there is none."""
    if not raw:
        return None
    runs = _HEX_RUN.findall(raw.strip().lower().replace("-", ""))
    return runs[-1][-32:] if runs else None  # "Facade-<id>": the ID is the last 32
