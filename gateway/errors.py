"""`{"error": {"code", "message"}}` bodies for every API error (ARCHITECTURE §10)."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from common.errors import ErrorCode
from common.logging import get_logger
from common.models import ErrorBody, ErrorInfo

log = get_logger()


class ApiError(Exception):
    def __init__(self, status_code: int, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_response(status_code: int, code: ErrorCode, message: str) -> JSONResponse:
    body = ErrorBody(error=ErrorInfo(code=code, message=message))
    return JSONResponse(body.model_dump(mode="json"), status_code=status_code)


async def _api_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return error_response(exc.status_code, exc.code, exc.message)


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Location and reason only — never the input, which may contain the Notion token.
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    message = f"{where}: {first.get('msg', 'invalid request')}" if where else "invalid request"
    return error_response(422, ErrorCode.VALIDATION_ERROR, message)


async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled_error", error=type(exc).__name__)
    return error_response(500, ErrorCode.INTERNAL_ERROR, "internal error")


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unexpected)
