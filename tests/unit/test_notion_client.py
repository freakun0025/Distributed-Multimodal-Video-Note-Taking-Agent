"""The Notion REST client against a mocked API (respx): limiter, headers, paging, errors."""

import json

import httpx
import pytest
import respx

from common.errors import ErrorCode, ExternalApi, PermanentError, RetryableError
from tests.unit.fakes import FakeLimiter
from worker.clients.notion import MAX_CHILDREN, NotionClient, notion_id

BASE = "https://notion.test"
API = f"{BASE}/v1"


def _client(limiter: FakeLimiter) -> NotionClient:
    return NotionClient("secret-token", base_url=BASE, version="2026-03-11", limiter=limiter,
                        timeout_s=5)  # fmt: skip


@respx.mock
async def test_every_call_takes_a_limiter_token_and_sends_the_version_and_token() -> None:
    route = respx.post(f"{API}/pages").respond(200, json={"id": "p1", "url": "https://n/p1"})
    limiter = FakeLimiter()
    async with _client(limiter) as notion:
        page = await notion.create_page("parent1", "Raft · #12345678")
    assert (page.id, page.url) == ("p1", "https://n/p1")
    assert limiter.calls == [(ExternalApi.NOTION, 1)]
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert request.headers["Notion-Version"] == "2026-03-11"
    assert json.loads(request.content) == {
        "parent": {"type": "page_id", "page_id": "parent1"},
        "properties": {
            "title": {"title": [{"type": "text", "text": {"content": "Raft · #12345678"}}]}
        },
    }


@respx.mock
async def test_child_pages_follow_every_page_of_the_listing() -> None:
    respx.get(f"{API}/blocks/parent1/children", params={"start_cursor": "c2"}).respond(
        200,
        json={"results": [{"id": "b", "type": "child_page", "child_page": {"title": "B"}}],
              "has_more": False, "next_cursor": None},
    )  # fmt: skip
    respx.get(f"{API}/blocks/parent1/children").respond(
        200,
        json={"results": [{"id": "a", "type": "child_page", "child_page": {"title": "A"}},
                          {"id": "x", "type": "paragraph", "paragraph": {}}],
              "has_more": True, "next_cursor": "c2"},
    )  # fmt: skip
    async with _client(FakeLimiter()) as notion:
        assert await notion.child_pages("parent1") == [("a", "A"), ("b", "B")]
        assert await notion.count_children("parent1") == 3


@respx.mock
async def test_append_sends_at_most_100_blocks_per_request_in_order() -> None:
    route = respx.patch(f"{API}/blocks/p1/children").respond(200, json={"results": []})
    blocks = [{"type": "paragraph", "n": i} for i in range(MAX_CHILDREN + 30)]
    async with _client(FakeLimiter()) as notion:
        await notion.append("p1", blocks)
    sent = [json.loads(call.request.content)["children"] for call in route.calls]
    assert [len(batch) for batch in sent] == [100, 30]
    assert [b["n"] for batch in sent for b in batch] == list(range(130))


@respx.mock
async def test_an_image_is_created_then_sent_as_multipart() -> None:
    respx.post(f"{API}/file_uploads").respond(200, json={"id": "up1", "status": "pending"})
    send = respx.post(f"{API}/file_uploads/up1/send").respond(200, json={"status": "uploaded"})
    limiter = FakeLimiter()
    async with _client(limiter) as notion:
        assert await notion.upload_image(b"JPEG", "frame_00020.jpg", "image/jpeg") == "up1"
    body = send.calls.last.request
    assert body.headers["content-type"].startswith("multipart/form-data")
    assert b'name="file"; filename="frame_00020.jpg"' in body.content and b"JPEG" in body.content
    assert len(limiter.calls) == 2


@pytest.mark.parametrize(
    ("status", "error", "code"),
    [
        (401, PermanentError, ErrorCode.NOTION_UNAUTHORIZED),
        (403, PermanentError, ErrorCode.NOTION_UNAUTHORIZED),
        (404, PermanentError, ErrorCode.NOTION_PARENT_NOT_FOUND),
        (409, RetryableError, ErrorCode.UPSTREAM_ERROR),
        (429, RetryableError, ErrorCode.RATE_LIMITED),
        (502, RetryableError, ErrorCode.UPSTREAM_ERROR),
    ],
)
@respx.mock
async def test_errors_are_classified_and_never_carry_the_token(
    status: int, error: type[Exception], code: ErrorCode
) -> None:
    respx.get(f"{API}/pages/p1").respond(
        status, json={"code": "some_code", "message": "Nope."}, headers={"Retry-After": "7"}
    )
    async with _client(FakeLimiter()) as notion:
        with pytest.raises(error) as raised:
            await notion.page("p1")
    assert raised.value.code is code  # type: ignore[attr-defined]
    assert "secret-token" not in str(raised.value)
    if status == 429:
        assert raised.value.retry_after_s == 7  # type: ignore[attr-defined]


@respx.mock
async def test_a_network_failure_is_retryable() -> None:
    respx.get(f"{API}/pages/p1").mock(side_effect=httpx.ConnectError("down"))
    async with _client(FakeLimiter()) as notion:
        with pytest.raises(RetryableError) as raised:
            await notion.page("p1")
    assert raised.value.code is ErrorCode.CONNECTION_ERROR


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0123456789abcdef0123456789abcdef", "0123456789abcdef0123456789abcdef"),
        ("01234567-89ab-cdef-0123-456789abcdef", "0123456789abcdef0123456789abcdef"),
        ("https://www.notion.so/me/Video-Notes-0123456789ABCDEF0123456789abcdef?pvs=4",
         "0123456789abcdef0123456789abcdef"),
        ("https://www.notion.so/Facade-0123456789abcdef0123456789abcdef",
         "0123456789abcdef0123456789abcdef"),
        ("  ", None),
        ("not an id", None),
        (None, None),
    ],
)  # fmt: skip
def test_a_pasted_page_id_or_link_becomes_the_bare_id(
    raw: str | None, expected: str | None
) -> None:
    assert notion_id(raw) == expected
