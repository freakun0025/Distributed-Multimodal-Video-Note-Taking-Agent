"""yt-dlp metadata/download and youtube-transcript-api wrappers (ARCHITECTURE §8.1, §8.3).

Both libraries block, so every call runs in a thread, and every call takes a `youtube` limiter
token first (invariant 6). Failures leave this module only as RetryableError / PermanentError.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import requests
import yt_dlp
from youtube_transcript_api import (
    AgeRestricted,
    FetchedTranscript,
    InvalidVideoId,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeRequestFailed,
    YouTubeTranscriptApi,
    YouTubeTranscriptApiException,
)
from yt_dlp.networking.exceptions import HTTPError as YtDlpHTTPError
from yt_dlp.networking.exceptions import TransportError as YtDlpTransportError
from yt_dlp.utils import DownloadError, ExtractorError, GeoRestrictedError

from common.config import Settings
from common.errors import (
    ErrorCode,
    ExternalApi,
    PermanentError,
    RetryableError,
    VnaError,
    classify_exception,
    classify_http,
)
from common.limiter import Limiter
from common.logging import get_logger
from common.models import Transcript, TranscriptSegment

if TYPE_CHECKING:
    from worker.clients.fakes import FakesYouTubeClient

API = ExternalApi.YOUTUBE
_MESSAGE_MAX_CHARS = 200
_PROGRESS_STEP = 0.05  # report download progress in 5% steps
log = get_logger()

ProgressFn = Callable[[float], Awaitable[None]]

# yt-dlp error texts (lower-cased) for cases it has no exception type for.
_BLOCKED_HINTS = ("not a bot", "too many requests", "http error 429")
_LIVE_HINTS = ("live event will begin", "premieres in", "premiere will begin")
_UNAVAILABLE_HINTS = (
    "private video",
    "unavailable",
    "removed",
    "terminated",
    "confirm your age",
    "age-restricted",
    "inappropriate",
    "members-only",
    "join this channel",
    "not available in your country",
    "copyright",
)


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def finished_download(dest_dir: Path, name: str) -> Path | None:
    """`<name>.<ext>` once a download is complete; partial files have a further suffix."""
    pattern = re.compile(rf"^{re.escape(name)}\.[A-Za-z0-9]+$")
    if not dest_dir.is_dir():
        return None
    return next((p for p in sorted(dest_dir.iterdir()) if pattern.match(p.name)), None)


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    title: str
    duration_sec: int | None
    live_status: str | None  # yt-dlp: not_live, was_live, is_live, is_upcoming, post_live
    channel: str | None
    language: str | None


class TranscriptTrack(Protocol):
    """One caption track in a youtube-transcript-api TranscriptList."""

    @property
    def language_code(self) -> str: ...

    @property
    def is_generated(self) -> bool: ...


class YouTubeClient:
    def __init__(self, limiter: Limiter, timeout_s: float) -> None:
        self._limiter = limiter
        self._timeout_s = timeout_s

    async def video_info(self, video_id: str) -> VideoInfo:
        """Title, duration and live status, without downloading anything."""
        await self._limiter.acquire(API)
        info = await asyncio.to_thread(self._extract_info, video_id)
        return video_info_from_ytdlp(video_id, info)

    async def transcript(self, video_id: str, languages: Sequence[str]) -> Transcript | None:
        """Captions in order of preference; None when the video has none usable."""
        await self._limiter.acquire(API)
        return await asyncio.to_thread(self._fetch_transcript, video_id, list(languages))

    async def download_video(
        self, video_id: str, dest_dir: Path, *, name: str, max_height: int, progress: ProgressFn
    ) -> Path:
        """Download once into `dest_dir/<name>.<ext>` (resumes a partial file after a crash)."""
        if (done := finished_download(dest_dir, name)) is not None:
            return done
        await self._limiter.acquire(API)
        loop = asyncio.get_running_loop()
        reports: list[concurrent.futures.Future[None]] = []

        async def forward(fraction: float) -> None:
            await progress(fraction)

        def report(fraction: float) -> None:  # called from yt-dlp's thread
            reports.append(asyncio.run_coroutine_threadsafe(forward(fraction), loop))

        try:
            return await asyncio.to_thread(
                self._download, video_id, dest_dir, name, max_height, report
            )
        finally:
            await asyncio.gather(*(asyncio.wrap_future(f) for f in reports), return_exceptions=True)

    def _download(
        self,
        video_id: str,
        dest_dir: Path,
        name: str,
        max_height: int,
        report: Callable[[float], None],
    ) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        reported = [0.0]

        def hook(status: dict[str, Any]) -> None:
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            if status.get("status") != "downloading" or not total:
                return
            fraction = min(status.get("downloaded_bytes", 0) / total, 1.0)
            if fraction - reported[0] >= _PROGRESS_STEP:
                reported[0] = fraction
                report(fraction)

        h = max_height
        options = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
            # Video only (frames need no audio); H.264 decodes fastest for ffmpeg seeks.
            "format": f"bv*[height<={h}][vcodec^=avc1]/bv*[height<={h}]/b[height<={h}]/b",
            "outtmpl": {"default": str(dest_dir / f"{name}.%(ext)s")},
            "socket_timeout": self._timeout_s,
            # A blip mid-download is retried once in place; anything else goes to the delay queues.
            "retries": 1,
            "fragment_retries": 1,
            "continuedl": True,
            "progress_hooks": [hook],
            "logger": _YtDlpLogger(),
        }
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.extract_info(watch_url(video_id), download=True)
        except Exception as exc:
            raise classify_ytdlp_error(exc) from exc
        path = finished_download(dest_dir, name)
        if path is None:
            raise RetryableError(ErrorCode.UPSTREAM_ERROR, "yt-dlp left no video file", api=API)
        return path

    def _extract_info(self, video_id: str) -> dict[str, Any]:
        options = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noplaylist": True,
            # No ignore_no_formats_error: with it, "This video is unavailable" becomes a warning
            # and an empty result instead of an error (seen against real YouTube, 2026-11-02).
            "socket_timeout": self._timeout_s,
            "logger": _YtDlpLogger(),
        }
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(watch_url(video_id), download=False, process=False)
        except Exception as exc:
            raise classify_ytdlp_error(exc) from exc
        if not isinstance(info, dict):
            raise RetryableError(ErrorCode.UPSTREAM_ERROR, "yt-dlp returned no metadata", api=API)
        return info

    def _fetch_transcript(self, video_id: str, languages: list[str]) -> Transcript | None:
        session = _TimeoutSession(self._timeout_s)
        try:
            listing = YouTubeTranscriptApi(http_client=session).list(video_id)
            track = choose_transcript(listing, languages)
            if track is None:
                return None
            fetched = track.fetch()
        except (TranscriptsDisabled, NoTranscriptFound):
            return None
        except Exception as exc:
            raise classify_transcript_error(exc) from exc
        finally:
            session.close()
        return transcript_from_fetched(fetched)


def video_source(settings: Settings, limiter: Limiter) -> YouTubeClient | FakesYouTubeClient:
    """Real YouTube, or the fakes service when FAKE_APIS=true."""
    if settings.fake_apis:
        from worker.clients.fakes import FakesYouTubeClient

        return FakesYouTubeClient(limiter, settings.fakes_url, settings.youtube_timeout_s)
    return YouTubeClient(limiter, settings.youtube_timeout_s)


def video_info_from_ytdlp(video_id: str, info: dict[str, Any]) -> VideoInfo:
    duration = info.get("duration")
    return VideoInfo(
        video_id=video_id,
        title=str(info.get("title") or f"YouTube video {video_id}"),
        # YouTube reports whole seconds; truncating keeps exactly 20:00 inside Constraint A.
        duration_sec=int(duration) if isinstance(duration, int | float) else None,
        live_status=info.get("live_status"),
        channel=info.get("channel") or info.get("uploader"),
        language=info.get("language"),
    )


def choose_transcript[T: TranscriptTrack](
    tracks: Iterable[T], languages: Sequence[str]
) -> T | None:
    """Manual in `languages` order → auto-generated in `languages` order → any manual → any."""
    available = list(tracks)
    for generated in (False, True):
        for code in languages:
            for track in available:
                if track.is_generated == generated and track.language_code == code:
                    return track
    manual = [track for track in available if not track.is_generated]
    candidates = manual or available
    return candidates[0] if candidates else None


def transcript_from_fetched(fetched: FetchedTranscript) -> Transcript | None:
    segments = [
        TranscriptSegment(start=max(s.start, 0.0), duration=max(s.duration, 0.0), text=text)
        for s in fetched.snippets
        if (text := " ".join(s.text.split()))
    ]
    if not segments:
        return None
    return Transcript(source="captions", language=fetched.language_code, segments=segments)


def classify_ytdlp_error(exc: BaseException) -> VnaError:
    for err in _causes(exc):
        if isinstance(err, VnaError):
            return err
        if isinstance(err, YtDlpHTTPError):
            return classify_http(API, err.status, detail=str(err))
        if isinstance(err, GeoRestrictedError):
            return RetryableError(ErrorCode.VIDEO_UNAVAILABLE, _short(err), api=API)
        if isinstance(err, YtDlpTransportError):
            if "timed out" in str(err).lower():
                return RetryableError(ErrorCode.UPSTREAM_TIMEOUT, "YouTube timed out", api=API)
            return RetryableError(ErrorCode.CONNECTION_ERROR, "YouTube connection failed", api=API)
    text = _short(exc)
    lowered = text.lower()
    if any(hint in lowered for hint in _BLOCKED_HINTS):
        return RetryableError(ErrorCode.RATE_LIMITED, text, api=API)
    if any(hint in lowered for hint in _LIVE_HINTS):
        return PermanentError(ErrorCode.VIDEO_UNSUPPORTED, text, api=API)
    if any(hint in lowered for hint in _UNAVAILABLE_HINTS):
        return RetryableError(ErrorCode.VIDEO_UNAVAILABLE, text, api=API)
    return RetryableError(ErrorCode.UPSTREAM_ERROR, text, api=API)


def classify_transcript_error(exc: BaseException) -> VnaError:
    if isinstance(exc, VideoUnavailable | VideoUnplayable | AgeRestricted):
        return RetryableError(ErrorCode.VIDEO_UNAVAILABLE, type(exc).__name__, api=API)
    if isinstance(exc, RequestBlocked):  # includes IpBlocked
        return RetryableError(
            ErrorCode.RATE_LIMITED, "YouTube is blocking transcript requests", api=API
        )
    if isinstance(exc, YouTubeRequestFailed):
        status = _http_status(exc)
        if status is not None:
            return classify_http(API, status, detail="transcript request")
        return RetryableError(ErrorCode.UPSTREAM_ERROR, "transcript request failed", api=API)
    if isinstance(exc, InvalidVideoId):
        return PermanentError(ErrorCode.UPSTREAM_REJECTED, "YouTube rejected the video id", api=API)
    if isinstance(exc, requests.Timeout):
        return RetryableError(ErrorCode.UPSTREAM_TIMEOUT, "YouTube timed out", api=API)
    if isinstance(exc, requests.ConnectionError):
        return RetryableError(ErrorCode.CONNECTION_ERROR, "YouTube connection failed", api=API)
    if isinstance(exc, YouTubeTranscriptApiException):  # PoTokenRequired, DataUnparsable, ...
        return RetryableError(ErrorCode.UPSTREAM_ERROR, type(exc).__name__, api=API)
    return classify_exception(API, exc)


def _causes(exc: BaseException) -> Iterator[BaseException]:
    """`exc` and what it wraps: DownloadError.exc_info, ExtractorError.cause, __cause__."""
    seen: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending:
        err = pending.pop(0)
        if err is None or id(err) in seen:
            continue
        seen.add(id(err))
        yield err
        if isinstance(err, DownloadError) and err.exc_info:
            pending.append(err.exc_info[1])
        if isinstance(err, ExtractorError) and isinstance(err.cause, BaseException):
            pending.append(err.cause)
        pending.extend([err.__cause__, err.__context__])


def _http_status(exc: YouTubeRequestFailed) -> int | None:
    for err in (exc.__cause__, exc.__context__):
        status = getattr(getattr(err, "response", None), "status_code", None)
        if isinstance(status, int):
            return status
    match = re.match(r"\s*(\d{3})\b", exc.reason)
    return int(match.group(1)) if match else None


def _short(exc: BaseException) -> str:
    text = str(exc)
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)  # yt-dlp colours its messages
    text = text.removeprefix("ERROR: ").strip()
    return text[:_MESSAGE_MAX_CHARS] or type(exc).__name__


class _TimeoutSession(requests.Session):
    """youtube-transcript-api passes no timeout; a hung request must not hang the worker."""

    def __init__(self, timeout_s: float) -> None:
        super().__init__()
        self._timeout_s = timeout_s

    def request(self, method: str, url: Any, *args: Any, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self._timeout_s)
        return super().request(method, url, *args, **kwargs)


class _YtDlpLogger:
    """Routes yt-dlp output into structlog at debug level instead of stderr."""

    def debug(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        log.debug("ytdlp_warning", message=message[:_MESSAGE_MAX_CHARS])

    def error(self, message: str) -> None:
        log.debug("ytdlp_error", message=message[:_MESSAGE_MAX_CHARS])
