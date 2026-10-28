import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from common.errors import WarningCode
from common.models import (
    FrameResult,
    JobCreateRequest,
    JobResult,
    JobResultsResponse,
    JobStatusResponse,
    JobWarning,
    StageMessage,
)
from common.state import JobStatus, Stage


def _message(**overrides: object) -> StageMessage:
    fields: dict[str, object] = {
        "job_id": uuid4(),
        "stage": Stage.TRANSCRIBE,
        "attempt": 1,
        "notion_token_enc": "gAAAAAB-ciphertext",
    }
    fields.update(overrides)
    return StageMessage.model_validate(fields)


def test_stage_message_round_trips_through_the_broker_body() -> None:
    message = _message(notion_parent_id="abc")
    assert StageMessage.from_body(message.to_body()) == message


def test_advancing_a_stage_resets_the_attempt_counter() -> None:
    message = _message(stage=Stage.SUMMARIZE, attempt=3)
    advanced = message.for_stage(Stage.EXTRACT_FRAMES)
    assert advanced.stage is Stage.EXTRACT_FRAMES
    assert advanced.attempt == 1
    assert advanced.job_id == message.job_id


def test_retry_increments_the_attempt_counter() -> None:
    assert _message(attempt=2).for_retry().attempt == 3


def test_stage_message_rejects_unknown_fields_and_bad_attempts() -> None:
    with pytest.raises(ValidationError):
        _message(notion_token="plaintext")  # only the encrypted field exists
    with pytest.raises(ValidationError):
        _message(attempt=0)


def test_stage_message_repr_hides_the_token() -> None:
    assert "gAAAAAB-ciphertext" not in repr(_message())


def test_job_request_keeps_the_token_secret_and_allows_it_missing() -> None:
    request = JobCreateRequest.model_validate(
        {"youtube_url": "https://youtu.be/dQw4w9WgXcQ", "notion_token": "secret_abc"}
    )
    assert "secret_abc" not in repr(request)
    assert request.notion_token is not None
    assert request.notion_token.get_secret_value() == "secret_abc"
    assert JobCreateRequest.model_validate({"youtube_url": "x"}).notion_token is None


def test_warning_serialises_exactly_like_the_spec_example() -> None:
    warning = JobWarning(code=WarningCode.FRAME_EXTRACTION_FAILED, timestamp_sec=461)
    assert json.loads(warning.model_dump_json()) == {
        "code": "FRAME_EXTRACTION_FAILED",
        "timestamp_sec": 461,
    }


def test_status_response_has_the_spec_fields_with_nulls() -> None:
    body = json.loads(
        JobStatusResponse(job_id=uuid4(), status=JobStatus.QUEUED, progress=0).model_dump_json()
    )
    for key in ("job_id", "status", "progress", "video_title", "duration_sec", "warnings"):
        assert key in body
    assert body["error"] is None
    assert body["result"] is None


def test_results_response_matches_the_spec_shape() -> None:
    response = JobResultsResponse(
        status=JobStatus.COMPLETED_WITH_WARNINGS,
        result=JobResult(
            notion_page_url="https://www.notion.so/page", frames_requested=6, frames_published=5
        ),
        warnings=[JobWarning(code=WarningCode.FRAME_EXTRACTION_FAILED, timestamp_sec=461)],
    )
    body = json.loads(response.model_dump_json())
    assert body["status"] == "COMPLETED_WITH_WARNINGS"
    assert body["result"] == {
        "notion_page_url": "https://www.notion.so/page",
        "frames_requested": 6,
        "frames_published": 5,
    }
    assert body["warnings"] == [{"code": "FRAME_EXTRACTION_FAILED", "timestamp_sec": 461}]


def test_frame_result_has_exactly_one_outcome() -> None:
    assert FrameResult(t_sec=10, path="frames/frame_00010.jpg").ok
    assert not FrameResult(t_sec=10, error="ffmpeg exited 1").ok
    with pytest.raises(ValidationError):
        FrameResult(t_sec=10)
    with pytest.raises(ValidationError):
        FrameResult(t_sec=10, path="p", error="e")
