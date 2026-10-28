"""YouTube URL -> 11-char video ID and canonical URL; pure (ARCHITECTURE §10, §15.2)."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

from common.errors import ErrorCode, PermanentError

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_MAX_URL_CHARS = 2048

_SHORT_HOSTS = frozenset({"youtu.be", "www.youtu.be"})
_LONG_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"})
# Path prefixes whose next segment is the video ID.
_ID_PATH_PREFIXES = frozenset({"shorts", "embed", "live", "v"})


def parse_video_id(url: str) -> str:
    """Return the video ID in `url`, or raise PermanentError(INVALID_URL).

    Accepts watch?v=, youtu.be/, /shorts/, /embed/, /live/, /v/ on www., m. and music. hosts,
    with or without a scheme, ignoring extra params such as &t= and &list=.
    """
    text = url.strip()
    if not text or len(text) > _MAX_URL_CHARS:
        raise _invalid("URL is empty or too long")
    if "://" not in text:
        text = f"https://{text}"

    parts = urlsplit(text)
    if parts.scheme.lower() not in ("http", "https"):
        raise _invalid("URL must use http or https")
    host = (parts.hostname or "").lower()
    segments = [segment for segment in parts.path.split("/") if segment]

    candidate: str | None = None
    if host in _SHORT_HOSTS:
        candidate = segments[0] if segments else None
    elif host in _LONG_HOSTS:
        if segments == ["watch"]:
            values = parse_qs(parts.query).get("v")
            candidate = values[0] if values else None
        elif len(segments) >= 2 and segments[0] in _ID_PATH_PREFIXES:
            candidate = segments[1]
    else:
        raise _invalid("not a YouTube URL")

    if candidate is None or not _VIDEO_ID.fullmatch(candidate):
        raise _invalid("no valid video ID in the URL")
    return candidate


def canonical_url(video_id: str) -> str:
    if not _VIDEO_ID.fullmatch(video_id):
        raise _invalid("invalid video ID")
    return f"https://www.youtube.com/watch?v={video_id}"


def _invalid(reason: str) -> PermanentError:
    return PermanentError(ErrorCode.INVALID_URL, f"Not a YouTube video link: {reason}.")
