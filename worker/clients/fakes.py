"""Clients for the fakes service, used when FAKE_APIS=true (ARCHITECTURE §14, D29).

Same interface and error classification as the real clients, so the fakes' failure switches
(429s, 5xx, a private video; CP9) take the same retry path; every call still takes a limiter
token first.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx

from common.errors import ErrorCode, RetryableError, VnaError, classify_exception, classify_http
from common.limiter import Limiter
from common.models import Transcript, TranscriptSegment
from worker.clients.youtube import (
    _PROGRESS_STEP,
    API,
    ProgressFn,
    VideoInfo,
    finished_download,
    video_info_from_ytdlp,
)


class FakesYouTubeClient:
    """FAKE_APIS=true: the same two calls against the fakes service (ARCHITECTURE §14).

    Real HTTP status codes from the fake go through classify_http, so injected 429s and 5xx
    take the same retry path as YouTube's, and a private video's 403 is VIDEO_UNAVAILABLE (D12).
    """

    def __init__(
        self,
        limiter: Limiter,
        base_url: str,
        timeout_s: float,
        http_client: httpx.AsyncClient | None = None,
        *,
        download_timeout_s: float | None = None,
    ) -> None:
        self._limiter = limiter
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._http_client = http_client
        self._download_timeout_s = download_timeout_s  # per download; None = no limit

    async def video_info(self, video_id: str) -> VideoInfo:
        data = await self._get(f"/youtube/videos/{video_id}")
        return video_info_from_ytdlp(video_id, data or {})

    async def transcript(self, video_id: str, languages: Sequence[str]) -> Transcript | None:
        data = await self._get(f"/youtube/videos/{video_id}/captions", missing_ok=True)
        if not data:
            return None
        segments = [
            TranscriptSegment(start=s["start"], duration=s["duration"], text=text)
            for s in data["segments"]
            if (text := " ".join(str(s["text"]).split()))
        ]
        if not segments:
            return None
        return Transcript(source="captions", language=data.get("language"), segments=segments)

    async def download_video(
        self, video_id: str, dest_dir: Path, *, name: str, max_height: int, progress: ProgressFn
    ) -> Path:
        if (done := finished_download(dest_dir, name)) is not None:
            return done
        await self._limiter.acquire(API)
        await asyncio.to_thread(dest_dir.mkdir, parents=True, exist_ok=True)
        final = dest_dir / f"{name}.mp4"
        part = dest_dir / f"{name}.mp4.{uuid.uuid4().hex}.part"
        url = f"{self._base_url}/youtube/videos/{video_id}/video"
        try:
            async with (
                asyncio.timeout(self._download_timeout_s),
                self._http() as http,
                http.stream("GET", url, params={"height": max_height}) as response,
            ):
                if response.status_code >= 400:
                    await response.aread()
                    raise classify_http(
                        API, response.status_code, response.headers, detail=_detail(response)
                    )
                total = int(response.headers.get("content-length") or 0)
                done_bytes, reported = 0, 0.0
                with part.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)
                        done_bytes += len(chunk)
                        if total and done_bytes / total - reported >= _PROGRESS_STEP:
                            reported = done_bytes / total
                            await progress(reported)
            os.replace(part, final)
        except VnaError:
            raise
        except TimeoutError:
            limit = self._download_timeout_s
            raise RetryableError(
                ErrorCode.UPSTREAM_TIMEOUT,
                f"the video download took longer than {limit:g} s",
                api=API,
            ) from None
        except Exception as exc:
            raise classify_exception(API, exc) from exc
        finally:
            part.unlink(missing_ok=True)
        return final

    @asynccontextmanager
    async def _http(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._http_client is not None:
            yield self._http_client
        else:
            async with httpx.AsyncClient(timeout=self._timeout_s) as http:
                yield http

    async def _get(self, path: str, *, missing_ok: bool = False) -> dict[str, Any] | None:
        await self._limiter.acquire(API)
        try:
            async with self._http() as http:
                response = await http.get(f"{self._base_url}{path}")
        except Exception as exc:
            raise classify_exception(API, exc) from exc
        if missing_ok and response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise classify_http(
                API, response.status_code, response.headers, detail=_detail(response)
            )
        data: dict[str, Any] = response.json()
        return data


def _detail(response: httpx.Response) -> str | None:
    """The fake's error message, like yt-dlp's ("Private video. Sign in ...")."""
    try:
        error = response.json().get("error")
    except (ValueError, AttributeError):
        return None
    message = error.get("message") if isinstance(error, dict) else None
    return str(message) if message else None
