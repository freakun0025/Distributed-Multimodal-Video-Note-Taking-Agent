"""YouTube client: caption choice, metadata mapping, error classification, limiter first.

No network: the blocking library calls are replaced; real YouTube runs only in the CP6 check.
"""

import io
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest
import requests
from youtube_transcript_api import (
    AgeRestricted,
    FetchedTranscript,
    FetchedTranscriptSnippet,
    InvalidVideoId,
    IpBlocked,
    PoTokenRequired,
    VideoUnavailable,
    YouTubeRequestFailed,
)
from yt_dlp.networking import Response
from yt_dlp.networking.exceptions import HTTPError, TransportError
from yt_dlp.utils import DownloadError, ExtractorError, GeoRestrictedError

from common.errors import ErrorCode, ExternalApi, RetryableError, VnaError
from common.models import Transcript, TranscriptSegment
from worker.clients import youtube
from worker.clients.youtube import (
    YouTubeClient,
    choose_transcript,
    classify_transcript_error,
    classify_ytdlp_error,
    transcript_from_fetched,
    video_info_from_ytdlp,
)

VIDEO_ID = "dQw4w9WgXcQ"


@dataclass
class Track:
    language_code: str
    is_generated: bool


LANGS = ["en", "en-US", "en-GB", "hi"]


@pytest.mark.parametrize(
    ("tracks", "expected"),
    [
        # A manual track in any preferred language beats an auto-generated preferred one.
        ([Track("en", True), Track("hi", False)], Track("hi", False)),
        # Preferred languages are tried in order.
        ([Track("en-GB", False), Track("en", False)], Track("en", False)),
        # Auto-generated in a preferred language beats a manual track in another language.
        ([Track("de", False), Track("en", True)], Track("en", True)),
        # Nothing preferred: any manual track, else any track.
        ([Track("ja", True), Track("de", False)], Track("de", False)),
        ([Track("ja", True)], Track("ja", True)),
        ([], None),
    ],
)
def test_caption_preference(tracks: list[Track], expected: Track | None) -> None:
    assert choose_transcript(tracks, LANGS) == expected


def _fetched(*snippets: tuple[str, float, float]) -> FetchedTranscript:
    return FetchedTranscript(
        snippets=[FetchedTranscriptSnippet(text=t, start=s, duration=d) for t, s, d in snippets],
        video_id=VIDEO_ID,
        language="English",
        language_code="en",
        is_generated=False,
    )


def test_caption_text_is_normalised_and_empty_lines_dropped() -> None:
    transcript = transcript_from_fetched(
        _fetched(("  Hello\n world ", 0.0, 2.5), ("   ", 2.5, 1.0), ("Raft  leaders", 3.5, 2.0))
    )
    assert transcript == Transcript(
        source="captions",
        language="en",
        segments=[
            TranscriptSegment(start=0.0, duration=2.5, text="Hello world"),
            TranscriptSegment(start=3.5, duration=2.0, text="Raft leaders"),
        ],
    )


def test_captions_with_no_text_count_as_no_transcript() -> None:
    assert transcript_from_fetched(_fetched(("\n", 0.0, 1.0))) is None


def test_metadata_mapping() -> None:
    info = video_info_from_ytdlp(
        VIDEO_ID,
        {"title": "Lecture 4", "duration": 1199.6, "live_status": "was_live", "uploader": "MIT"},
    )
    assert (info.title, info.duration_sec, info.live_status, info.channel) == (
        "Lecture 4",
        1199,
        "was_live",
        "MIT",
    )
    unknown = video_info_from_ytdlp(VIDEO_ID, {})
    assert (unknown.title, unknown.duration_sec) == (f"YouTube video {VIDEO_ID}", None)


def _download_error(message: str, cause: BaseException | None = None) -> DownloadError:
    """What YoutubeDL.extract_info raises: a coloured DownloadError wrapping the ExtractorError."""
    try:
        raise ExtractorError(message, expected=True, cause=cause)
    except ExtractorError:
        text = f"ERROR: \x1b[0;31m[youtube] {VIDEO_ID}: {message}\x1b[0m"
        return DownloadError(text, sys.exc_info())


def _http_error(status: int) -> HTTPError:
    return HTTPError(Response(io.BytesIO(b""), "https://www.youtube.com", {}, status=status))


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (
            _download_error("Private video. Sign in if you've been granted access"),
            "VIDEO_UNAVAILABLE",
            True,
        ),
        (
            _download_error("Video unavailable. This video has been removed"),
            "VIDEO_UNAVAILABLE",
            True,
        ),
        (_download_error("Sign in to confirm your age"), "VIDEO_UNAVAILABLE", True),
        # YouTube writes this with a typographic apostrophe.
        (_download_error("Sign in to confirm you’re not a bot"), "RATE_LIMITED", True),  # noqa: RUF001
        (_download_error("This live event will begin in 3 hours."), "VIDEO_UNSUPPORTED", False),
        (_download_error("Unable to download webpage", _http_error(429)), "RATE_LIMITED", True),
        (_download_error("Unable to download webpage", _http_error(503)), "UPSTREAM_ERROR", True),
        (
            _download_error("Unable to download API page", TransportError("timed out")),
            "UPSTREAM_TIMEOUT",
            True,
        ),
        (
            _download_error("Unable to download webpage", TransportError("refused")),
            "CONNECTION_ERROR",
            True,
        ),
        (GeoRestrictedError("not available from your location"), "VIDEO_UNAVAILABLE", True),
        (_download_error("Something new and strange"), "UPSTREAM_ERROR", True),
    ],
)
def test_ytdlp_errors_are_classified(error: Exception, code: str, retryable: bool) -> None:
    classified = classify_ytdlp_error(error)
    assert (classified.code.value, classified.retryable) == (code, retryable)
    assert classified.api is ExternalApi.YOUTUBE


def test_ytdlp_messages_are_short_and_uncoloured() -> None:
    classified = classify_ytdlp_error(_download_error("Private video. " + "x" * 500))
    assert "\x1b" not in classified.message
    assert not classified.message.startswith("ERROR:")
    assert len(classified.message) <= 200


def _request_failed(status: int) -> YouTubeRequestFailed:
    response = requests.Response()
    response.status_code = status
    try:
        raise requests.HTTPError(
            f"{status} Client Error: x for url: https://youtube", response=response
        )
    except requests.HTTPError as http_error:
        try:
            raise YouTubeRequestFailed(
                VIDEO_ID, http_error
            )  # implicit chaining, as the library does
        except YouTubeRequestFailed as failed:
            return failed


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (VideoUnavailable(VIDEO_ID), "VIDEO_UNAVAILABLE", True),
        (AgeRestricted(VIDEO_ID), "VIDEO_UNAVAILABLE", True),
        (IpBlocked(VIDEO_ID), "RATE_LIMITED", True),
        (_request_failed(429), "RATE_LIMITED", True),
        (_request_failed(503), "UPSTREAM_ERROR", True),
        (_request_failed(404), "VIDEO_UNAVAILABLE", True),
        (
            YouTubeRequestFailed(VIDEO_ID, requests.HTTPError("connection reset")),
            "UPSTREAM_ERROR",
            True,
        ),
        (PoTokenRequired(VIDEO_ID), "UPSTREAM_ERROR", True),
        (InvalidVideoId(VIDEO_ID), "UPSTREAM_REJECTED", False),
        (requests.ConnectTimeout("slow"), "UPSTREAM_TIMEOUT", True),
        (requests.ConnectionError("refused"), "CONNECTION_ERROR", True),
        (RuntimeError("bug"), "INTERNAL_ERROR", True),
    ],
)
def test_transcript_errors_are_classified(error: Exception, code: str, retryable: bool) -> None:
    classified = classify_transcript_error(error)
    assert (classified.code.value, classified.retryable) == (code, retryable)


class OrderedLimiter:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def acquire(self, api: ExternalApi, cost: float = 1) -> float:
        self.events.append(f"acquire:{api.value}")
        return 0.0


async def test_every_youtube_call_takes_a_limiter_token_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    client = YouTubeClient(OrderedLimiter(events), timeout_s=5)

    def extract(video_id: str) -> dict[str, Any]:
        events.append("metadata")
        return {"title": "T", "duration": 60}

    def fetch(video_id: str, languages: list[str]) -> None:
        events.append("captions")

    monkeypatch.setattr(client, "_extract_info", extract)
    monkeypatch.setattr(client, "_fetch_transcript", fetch)
    await client.video_info(VIDEO_ID)
    await client.transcript(VIDEO_ID, LANGS)
    assert events == ["acquire:youtube", "metadata", "acquire:youtube", "captions"]


async def test_a_refused_limiter_token_means_no_youtube_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Refusing:
        async def acquire(self, api: ExternalApi, cost: float = 1) -> float:
            raise RetryableError(ErrorCode.RATE_LIMITED_LOCAL, "busy", api=api, retry_after_s=30)

    client = YouTubeClient(Refusing(), timeout_s=5)
    monkeypatch.setattr(client, "_extract_info", lambda video_id: pytest.fail("called YouTube"))
    with pytest.raises(VnaError) as caught:
        await client.video_info(VIDEO_ID)
    assert caught.value.code is ErrorCode.RATE_LIMITED_LOCAL


def test_transcript_requests_get_a_timeout_unless_one_is_given() -> None:
    session = youtube._TimeoutSession(7)
    with patch.object(requests.Session, "request", return_value="ok") as parent:
        session.request("GET", "https://www.youtube.com/watch?v=x")
        session.request("GET", "https://www.youtube.com/api", timeout=2)
    assert [c.kwargs["timeout"] for c in parent.call_args_list] == [7, 2]
