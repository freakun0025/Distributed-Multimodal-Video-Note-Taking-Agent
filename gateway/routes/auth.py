"""Notion OAuth for the extension (ARCHITECTURE §10, §11; D10): the gateway swaps the code
for a token because the client secret must never ship in an extension. It stores no tokens and
never logs them; every call to Notion takes a `notion` limiter token first (invariant 6).

GET  /auth/notion/config    -> {client_id, redirect_uri} to build the authorize URL
POST /auth/notion/exchange  -> {code} -> token pair
POST /auth/notion/refresh   -> {refresh_token} -> new token pair
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, Request

from common.config import Settings
from common.errors import ErrorCode, ExternalApi, RetryableError, classify_exception
from common.logging import get_logger
from common.models import (
    ErrorBody,
    NotionClientConfig,
    NotionExchangeRequest,
    NotionRefreshRequest,
    NotionTokenResponse,
)
from gateway.errors import ApiError

router = APIRouter(prefix="/auth/notion")
log = get_logger()
API = ExternalApi.NOTION

_ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorBody, "description": "OAUTH_CODE_INVALID"},
    429: {"model": ErrorBody, "description": "RATE_LIMITED"},
    502: {"model": ErrorBody, "description": "UPSTREAM_ERROR"},
}


@router.get("/config", response_model=NotionClientConfig)
async def config(request: Request) -> NotionClientConfig:
    settings: Settings = request.app.state.settings
    if not settings.notion_client_id or not settings.notion_redirect_uri:
        raise ApiError(500, ErrorCode.INTERNAL_ERROR, "Notion OAuth is not configured")
    return NotionClientConfig(
        client_id=settings.notion_client_id, redirect_uri=settings.notion_redirect_uri
    )


@router.post("/exchange", response_model=NotionTokenResponse, responses=_ERRORS)
async def exchange(body: NotionExchangeRequest, request: Request) -> NotionTokenResponse:
    settings: Settings = request.app.state.settings
    grant = {
        "grant_type": "authorization_code",
        "code": body.code,
        "redirect_uri": settings.notion_redirect_uri,
    }
    token = await _token(request, grant, rejected="Notion did not accept the sign-in code")
    log.info("notion_oauth_connected")
    return token


@router.post("/refresh", response_model=NotionTokenResponse, responses=_ERRORS)
async def refresh(body: NotionRefreshRequest, request: Request) -> NotionTokenResponse:
    grant = {"grant_type": "refresh_token", "refresh_token": body.refresh_token.get_secret_value()}
    token = await _token(request, grant, rejected="Notion did not accept the refresh token")
    log.info("notion_oauth_refreshed")
    return token


async def _token(request: Request, grant: dict[str, Any], *, rejected: str) -> NotionTokenResponse:
    """POST /v1/oauth/token with HTTP Basic client credentials (Notion's OAuth docs)."""
    settings: Settings = request.app.state.settings
    secret = settings.notion_client_secret
    if not settings.notion_client_id or secret is None or not settings.notion_redirect_uri:
        raise ApiError(500, ErrorCode.INTERNAL_ERROR, "Notion OAuth is not configured")
    base = f"{settings.fakes_url}/notion" if settings.fake_apis else settings.notion_api_url
    try:
        await request.app.state.limiter.acquire(API)
    except RetryableError as exc:
        if exc.code is ErrorCode.RATE_LIMITED_LOCAL:  # the wait would be too long
            raise ApiError(
                429, ErrorCode.RATE_LIMITED, "too many requests to Notion; try again"
            ) from exc
        raise ApiError(503, exc.code, "the rate limiter is unavailable; try again") from exc
    try:
        async with httpx.AsyncClient(timeout=settings.notion_timeout_s) as http:
            response = await http.post(
                f"{base.rstrip('/')}/v1/oauth/token",
                json=grant,
                auth=httpx.BasicAuth(settings.notion_client_id, secret.get_secret_value()),
                headers={"Notion-Version": settings.notion_api_version},
            )
    except Exception as exc:
        error = classify_exception(API, exc)
        log.warning("notion_oauth_unreachable", code=error.code)
        raise ApiError(502, ErrorCode.UPSTREAM_ERROR, "Notion could not be reached") from exc
    if response.status_code == 200:
        data = response.json()
        return NotionTokenResponse(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            workspace_name=data.get("workspace_name"),
            bot_id=data.get("bot_id"),
            expires_in=data.get("expires_in"),
        )
    notion_code = _notion_error(response)
    log.warning("notion_oauth_rejected", status=response.status_code, notion_code=notion_code)
    if response.status_code == 429:
        raise ApiError(429, ErrorCode.RATE_LIMITED, "Notion is rate limiting; try again shortly")
    if response.status_code >= 500:
        raise ApiError(502, ErrorCode.UPSTREAM_ERROR, "Notion had a problem; try again")
    if response.status_code == 401 or notion_code == "invalid_client":
        # Our client ID/secret, not the user's code: a configuration problem on the server.
        raise ApiError(
            500, ErrorCode.INTERNAL_ERROR, "Notion rejected the server's client credentials"
        )
    raise ApiError(400, ErrorCode.OAUTH_CODE_INVALID, rejected)


def _notion_error(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    code = body.get("error") or body.get("code")
    return str(code) if code else None
