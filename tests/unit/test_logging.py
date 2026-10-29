import json
import logging

import pytest
import structlog
from pydantic import SecretStr

from common.logging import REDACTED, bind_job, clear_job, configure_logging, redact_secrets


def _redact(**event: object) -> dict[str, object]:
    return dict(redact_secrets(None, "info", dict(event)))


@pytest.mark.parametrize(
    "key",
    [
        "notion_token",
        "access_token",
        "refresh_token",
        "notion_token_enc",
        "client_secret",
        "notion_client_secret",
        "llm_api_key",
        "token_encryption_key",
        "Authorization",
        "oauth_code",
        "password",
    ],
)
def test_secret_keys_are_redacted(key: str) -> None:
    assert _redact(**{key: "value"})[key] == REDACTED


def test_ordinary_fields_including_error_codes_survive() -> None:
    event = _redact(event="x", code="RATE_LIMITED", job_id="j", stage="PUBLISH", attempt=2)
    assert event == {
        "event": "x",
        "code": "RATE_LIMITED",
        "job_id": "j",
        "stage": "PUBLISH",
        "attempt": 2,
    }


def test_nested_secrets_and_secretstr_values_are_redacted() -> None:
    event = _redact(
        request={"headers": {"Authorization": "Bearer abc"}, "body": {"notion_token": "t"}},
        items=[{"refresh_token": "r"}],
        key_value=SecretStr("hidden"),
    )
    assert event["request"] == {
        "headers": {"Authorization": REDACTED},
        "body": {"notion_token": REDACTED},
    }
    assert event["items"] == [{"refresh_token": REDACTED}]
    assert event["key_value"] == REDACTED


# Fake secrets are assembled at runtime so no secret-shaped literal sits in the source:
# GitHub secret scanning flags such literals even when they are obviously fake.
@pytest.mark.parametrize(
    "secret",
    [
        "ntn" + "_" + "x" * 20,
        "secret" + "_" + "x" * 16,
        "AI" + "za" + "x" * 35,
        "Bearer " + "x" * 24,
        "gAAAA" + "A" + "x" * 30,
    ],
)
def test_secret_looking_text_inside_messages_is_scrubbed(secret: str) -> None:
    event = _redact(event=f"call failed with {secret} attached")
    assert secret not in str(event["event"])
    assert REDACTED in str(event["event"])


def test_configured_logger_emits_json_with_job_context(capsys: pytest.CaptureFixture[str]) -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    configure_logging("INFO", service="test")
    bind_job("job-1", stage="SUMMARIZE", attempt=3)
    try:
        structlog.get_logger().info("stage_started", notion_token="leak")
        logging.getLogger("uvicorn.error").info("stdlib line")
    finally:
        clear_job()
        structlog.reset_defaults()
        root.handlers, root.level = saved_handlers, saved_level
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    ours, stdlib = lines[0], lines[1]
    assert ours["event"] == "stage_started"
    assert ours["job_id"] == "job-1"
    assert ours["stage"] == "SUMMARIZE"
    assert ours["attempt"] == 3
    assert ours["notion_token"] == REDACTED
    assert ours["service"] == "test"
    assert stdlib["event"] == "stdlib line"


def test_logged_exceptions_carry_no_locals_and_scrubbed_messages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    token = "ntn" + "_" + "y" * 24

    def call_notion(notion_token: str) -> None:
        raise RuntimeError(f"request failed for {notion_token}")

    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    configure_logging("INFO", service="test")
    try:
        try:
            call_notion(token)
        except RuntimeError:
            structlog.get_logger().exception("stage_crashed")
            logging.getLogger("aio_pika").exception("stdlib crash")
    finally:
        structlog.reset_defaults()
        root.handlers, root.level = saved_handlers, saved_level
    out = capsys.readouterr().out
    lines = [json.loads(line) for line in out.splitlines() if line]
    assert [line["event"] for line in lines] == ["stage_crashed", "stdlib crash"]
    assert all("exception" in line for line in lines)  # tracebacks are still there
    assert token not in out
    assert '"locals"' not in out
