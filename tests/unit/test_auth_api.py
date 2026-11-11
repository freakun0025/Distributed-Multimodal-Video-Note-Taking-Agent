"""Notion OAuth on the gateway (ARCHITECTURE §10, §11; D10): code -> token, nothing stored."""

import base64
import json
from typing import Any

import httpx
import pytest
import respx

from common.config import Settings
from common.errors import ExternalApi
from gateway.main import create_app
from tests.unit.fakes import FakeLimiter

TOKEN_URL = "https://notion.test/v1/oauth/token"
REDIRECT = "https://hiimaheehbfpmgnjjpeoejngcmhjmjfi.chromiumapp.org/notion"
CONFIGURED = {
    "notion_client_id": "client-1",
    "notion_client_secret": "very-secret",
    "notion_redirect_uri": REDIRECT,
    "notion_api_url": "https://notion.test",
}
NOTION_TOKEN = {
    "access_token": "ntn_access",
    "token_type": "bearer",
    "refresh_token": "ntn_refresh",
    "bot_id": "bot-1",
    "workspace_id": "ws-1",
    "workspace_name": "My workspace",
    "workspace_icon": None,
    "owner": {"type": "workspace", "workspace": True},
    "duplicated_template_id": None,
}


async def _request(
    method: str, path: str, *, limiter: FakeLimiter | None = None, **kwargs: Any
) -> httpx.Response:
    settings = kwargs.pop("settings", CONFIGURED)
    app = create_app(Settings(_env_file=None, **settings))  # type: ignore[call-arg]
    app.state.limiter = limiter or FakeLimiter()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


async def test_config_gives_the_extension_the_public_client_id_and_redirect() -> None:
    response = await _request("GET", "/auth/notion/config")
    assert response.status_code == 200
    assert response.json() == {"client_id": "client-1", "redirect_uri": REDIRECT}
    assert "very-secret" not in response.text


async def test_without_notion_settings_the_gateway_says_so() -> None:
    response = await _request("GET", "/auth/notion/config", settings={})
    assert response.status_code == 500
    assert response.json()["error"]["message"] == "Notion OAuth is not configured"


@respx.mock
async def test_the_code_is_exchanged_with_the_secret_and_the_token_returned() -> None:
    route = respx.post(TOKEN_URL).respond(200, json=NOTION_TOKEN)
    limiter = FakeLimiter()
    response = await _request(
        "POST", "/auth/notion/exchange", json={"code": "the-code"}, limiter=limiter
    )
    assert response.status_code == 200
    assert response.json() == {
        "access_token": "ntn_access",
        "refresh_token": "ntn_refresh",
        "workspace_name": "My workspace",
        "bot_id": "bot-1",
        "expires_in": None,
    }
    sent = route.calls.last.request
    assert (
        sent.headers["Authorization"]
        == "Basic " + base64.b64encode(b"client-1:very-secret").decode()
    )
    assert sent.headers["Notion-Version"] == "2026-03-11"
    assert json.loads(sent.content) == {
        "grant_type": "authorization_code",
        "code": "the-code",
        "redirect_uri": REDIRECT,
    }
    assert limiter.calls == [(ExternalApi.NOTION, 1)]


@respx.mock
async def test_a_refresh_token_gets_a_new_pair() -> None:
    route = respx.post(TOKEN_URL).respond(200, json=NOTION_TOKEN)
    response = await _request("POST", "/auth/notion/refresh", json={"refresh_token": "old"})
    assert response.status_code == 200 and response.json()["access_token"] == "ntn_access"
    assert json.loads(route.calls.last.request.content) == {
        "grant_type": "refresh_token",
        "refresh_token": "old",
    }


@pytest.mark.parametrize(
    ("status", "body", "expected", "code"),
    [
        (400, {"error": "invalid_grant"}, 400, "OAUTH_CODE_INVALID"),
        (401, {"error": "invalid_client"}, 500, "INTERNAL_ERROR"),
        (429, {}, 429, "RATE_LIMITED"),
        (503, {}, 502, "UPSTREAM_ERROR"),
    ],
)
@respx.mock
async def test_notion_failures_become_clear_errors_without_secrets(
    status: int, body: dict[str, Any], expected: int, code: str
) -> None:
    respx.post(TOKEN_URL).respond(status, json=body)
    response = await _request("POST", "/auth/notion/exchange", json={"code": "the-code"})
    assert response.status_code == expected
    assert response.json()["error"]["code"] == code
    assert "very-secret" not in response.text and "the-code" not in response.text


@respx.mock
async def test_notion_unreachable_is_a_502() -> None:
    respx.post(TOKEN_URL).mock(side_effect=httpx.ConnectError("down"))
    response = await _request("POST", "/auth/notion/exchange", json={"code": "c"})
    assert response.status_code == 502 and response.json()["error"]["code"] == "UPSTREAM_ERROR"


async def test_an_empty_code_is_a_validation_error() -> None:
    response = await _request("POST", "/auth/notion/exchange", json={"code": ""})
    assert response.status_code == 422
