"""structlog JSON logs bound to job_id/stage/attempt, secrets redacted (ARCHITECTURE §13)."""

from __future__ import annotations

import logging
import re
import socket
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

import structlog
from pydantic import SecretStr
from structlog.typing import EventDict, FilteringBoundLogger, Processor, WrappedLogger

REDACTED = "[REDACTED]"

_SECRET_KEYS = frozenset(
    {"token", "secret", "password", "authorization", "api_key", "cookie", "oauth_code"}
)
_SECRET_SUFFIXES = ("_token", "_token_enc", "_secret", "_api_key", "_encryption_key", "_password")
# Secret-looking substrings inside free text: Notion tokens, Google API keys, bearer headers,
# Fernet ciphertexts.
_SECRET_PATTERNS = (
    re.compile(r"\b(?:ntn|secret)_[A-Za-z0-9]{10,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]+=*"),
    re.compile(r"\bgAAAAA[A-Za-z0-9_\-]{20,}=*"),
)


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _SECRET_KEYS or lowered.endswith(_SECRET_SUFFIXES)


def _scrub(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return REDACTED
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub(REDACTED, value)
        return value
    if isinstance(value, Mapping):
        return {
            key: REDACTED if isinstance(key, str) and _is_secret_key(key) else _scrub(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return type(value)(_scrub(item) for item in value)
    return value


def redact_secrets(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Never let tokens, OAuth codes, client secrets or API keys reach a log line (CLAUDE.md §5)."""
    for key in list(event_dict):
        event_dict[key] = REDACTED if _is_secret_key(key) else _scrub(event_dict[key])
    return event_dict


def _static_fields(service: str) -> Processor:
    worker = socket.gethostname()

    def add(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
        event_dict.setdefault("service", service)
        event_dict.setdefault("worker", worker)
        return event_dict

    return add


def configure_logging(level: str = "INFO", *, service: str) -> None:
    """JSON lines on stdout for our loggers and for stdlib ones (uvicorn, aio-pika, alembic)."""
    numeric_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _static_fields(service),
        redact_secrets,
    ]
    structlog.configure(
        processors=[
            *shared,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.WriteLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(numeric_level)


def get_logger(**initial: Any) -> FilteringBoundLogger:
    logger: FilteringBoundLogger = structlog.get_logger(**initial)
    return logger


def bind_job(job_id: str, stage: str | None = None, attempt: int | None = None) -> None:
    """Bind job context for every log line in the current task (ARCHITECTURE §13)."""
    fields: MutableMapping[str, Any] = {"job_id": job_id}
    if stage is not None:
        fields["stage"] = stage
    if attempt is not None:
        fields["attempt"] = attempt
    structlog.contextvars.bind_contextvars(**fields)


def clear_job() -> None:
    structlog.contextvars.clear_contextvars()
