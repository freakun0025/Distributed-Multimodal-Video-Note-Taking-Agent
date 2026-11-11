"""httpx Notion REST client: pages, children, file uploads (ARCHITECTURE §8.4).

Every call takes a `notion` limiter token first (invariant 6). Failures go through
classify_http, so 429/5xx/409 retry through the delay queues (Retry-After honoured) and
401/403/404 fail the job. The token is only ever in the Authorization header, never logged.
Pages are found by listing the parent's children, never with the search API (invariant 9).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol

import httpx

from common.config import Settings
from common.errors import ExternalApi, VnaError, classify_exception, classify_http
from common.limiter import Limiter

API = ExternalApi.NOTION
MAX_CHILDREN = 100  # blocks per append request (Notion request limits)
_PAGE_SIZE = 100
_HEX_RUN = re.compile(r"[0-9a-f]{32,}")


@dataclass(frozen=True)
class NotionPage:
    id: str
    url: str


def notion_id(raw: str | None) -> str | None:
    """A page ID from what a user pastes: the bare ID, the dashed UUID or the page's URL
    (whose last 32 hex characters are the ID); None if there is none."""
    if not raw:
        return None
    runs = _HEX_RUN.findall(raw.strip().lower().replace("-", ""))
    return runs[-1][-32:] if runs else None  # "Facade-<id>": the ID is the last 32


class NotionApi(Protocol):
    """The Notion calls PUBLISH makes (NotionClient; an in-memory double in tests)."""

    async def child_pages(self, parent_id: str) -> list[tuple[str, str]]: ...
    async def count_children(self, block_id: str) -> int: ...
    async def page(self, page_id: str) -> NotionPage: ...
    async def create_page(self, parent_id: str, title: str) -> NotionPage: ...
    async def append(self, block_id: str, children: Sequence[Mapping[str, Any]]) -> None: ...
    async def upload_image(self, data: bytes, filename: str, content_type: str) -> str: ...


class NotionClient:
    """One job's Notion calls with the user's token. Use as `async with`."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str,
        version: str,
        limiter: Limiter,
        timeout_s: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = base_url.rstrip("/") + "/v1"
        self._headers = {"Authorization": f"Bearer {token}", "Notion-Version": version}
        self._limiter = limiter
        self._timeout_s = timeout_s
        self._given = http_client
        self._http: httpx.AsyncClient | None = http_client

    async def __aenter__(self) -> NotionClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout_s)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._given is None and self._http is not None:
            await self._http.aclose()
            self._http = None

    async def child_pages(self, parent_id: str) -> list[tuple[str, str]]:
        """(id, title) of every child page of `parent_id`, in order."""
        return [
            (block["id"], str((block.get("child_page") or {}).get("title", "")))
            for block in await self._children(parent_id)
            if block.get("type") == "child_page"
        ]

    async def count_children(self, block_id: str) -> int:
        """Top-level blocks already on a page (resume-safe append: append blocks[k:])."""
        return len(await self._children(block_id))

    async def page(self, page_id: str) -> NotionPage:
        data = await self._call("GET", f"/pages/{page_id}")
        return NotionPage(id=data["id"], url=data["url"])

    async def create_page(self, parent_id: str, title: str) -> NotionPage:
        """An empty page under a parent page, titled `title`."""
        body = {
            "parent": {"type": "page_id", "page_id": parent_id},
            "properties": {"title": {"title": [{"type": "text", "text": {"content": title}}]}},
        }
        data = await self._call("POST", "/pages", json=body)
        return NotionPage(id=data["id"], url=data["url"])

    async def append(self, block_id: str, children: Sequence[Mapping[str, Any]]) -> None:
        """Append in order, at most MAX_CHILDREN per request."""
        for start in range(0, len(children), MAX_CHILDREN):
            batch = [dict(child) for child in children[start : start + MAX_CHILDREN]]
            await self._call("PATCH", f"/blocks/{block_id}/children", json={"children": batch})

    async def upload_image(self, data: bytes, filename: str, content_type: str) -> str:
        """Direct upload (create, then send): the file upload's ID, to attach within the hour
        (Notion archives unattached uploads), so it is never stored across attempts."""
        created = await self._call(
            "POST", "/file_uploads", json={"filename": filename, "content_type": content_type}
        )
        upload_id = str(created["id"])
        await self._call(
            "POST",
            f"/file_uploads/{upload_id}/send",
            files={"file": (filename, data, content_type)},
        )
        return upload_id

    async def _children(self, block_id: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"page_size": _PAGE_SIZE}
            if cursor:
                params["start_cursor"] = cursor
            data = await self._call("GET", f"/blocks/{block_id}/children", params=params)
            blocks.extend(data.get("results") or [])
            cursor = data.get("next_cursor")
            if not data.get("has_more") or not cursor:
                return blocks

    async def _call(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        files: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._http is None:
            raise RuntimeError("NotionClient used outside `async with`")
        await self._limiter.acquire(API)
        try:
            response = await self._http.request(
                method, self._base + path, headers=self._headers, json=json, params=params,
                files=files,
            )  # fmt: skip
        except VnaError:
            raise
        except Exception as exc:
            raise classify_exception(API, exc) from exc
        if response.status_code >= 400:
            raise classify_http(
                API, response.status_code, response.headers, detail=_detail(response)
            )
        data: dict[str, Any] = response.json()
        return data


def notion_client(settings: Settings, limiter: Limiter, token: str) -> NotionClient:
    """The real API, or the fakes service when FAKE_APIS=true (ARCHITECTURE §14)."""
    base_url = f"{settings.fakes_url}/notion" if settings.fake_apis else settings.notion_api_url
    return NotionClient(
        token,
        base_url=base_url,
        version=settings.notion_api_version,
        limiter=limiter,
        timeout_s=settings.notion_timeout_s,
    )


def _detail(response: httpx.Response) -> str | None:
    """Notion's error code and message (never request data, so never the token)."""
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    code, message = body.get("code"), body.get("message")
    return f"{code}: {message}" if code or message else None
