"""chaos/burst.py's report: LLM answers read from the workers' logs, and the verdict (Spec §9)."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from chaos.burst import JobRow, LlmAnswer, parse_llm_answers, report, video_ids
from common.state import JobStatus
from common.youtube_url import parse_video_id

JOB = str(uuid4())
OTHER = str(uuid4())
REAL = (
    '{"event": "HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/'
    'gemini-3.6-flash:generateContent \\"HTTP/1.1 429 Too Many Requests\\"", '
    f'"job_id": "{JOB}", "level": "info"}}'
)
FAKE = (
    '{"event": "HTTP Request: POST http://fakes:8090/llm/v1beta/models/'
    'gemini-3.8-flash:generateContent \\"HTTP/1.1 200 OK\\"", '
    f'"job_id": "{JOB}", "level": "info"}}'
)


def test_every_llm_answer_is_read_from_the_worker_logs() -> None:
    lines = [
        REAL,
        "vna-worker-2  | " + FAKE,  # with docker's prefix
        FAKE.replace(JOB, OTHER),  # another job's call
        '{"event": "HTTP Request: GET http://fakes:8090/notion/v1/pages \\"HTTP/1.1 200 OK\\"", '
        f'"job_id": "{JOB}"}}',  # not the LLM
        "not json at all",
    ]
    assert parse_llm_answers(lines, {JOB}) == [
        LlmAnswer(JOB, "gemini-3.6-flash", 429),
        LlmAnswer(JOB, "gemini-3.8-flash", 200),
    ]


def _row(status: JobStatus, after_s: float | None) -> JobRow:
    created = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
    return JobRow(
        id=uuid4(),
        video_id="burst0a1b2c",
        title="Lecture",
        status=status,
        created_at=created,
        finished_at=created + timedelta(seconds=after_s) if after_s is not None else None,
        error_code=None,
    )


@pytest.mark.parametrize(
    ("statuses", "answers", "ok"),
    [
        ([JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS], [200, 200], True),
        ([JobStatus.COMPLETED, JobStatus.COMPLETED], [200, 429, 200], False),  # one 429
        ([JobStatus.COMPLETED, JobStatus.DEAD_LETTERED], [200], False),
        ([JobStatus.COMPLETED, JobStatus.SUMMARIZING], [200], False),  # not drained
    ],
)
def test_a_burst_passes_only_when_every_job_completes_without_an_llm_429(
    statuses: list[JobStatus], answers: list[int], ok: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        _row(status, 90.0 + i if status not in {JobStatus.SUMMARIZING} else None)
        for i, status in enumerate(statuses)
    ]
    assert report(rows, [LlmAnswer(JOB, "m", s) for s in answers]) is ok
    out = capsys.readouterr().out
    assert f"429s from the LLM: {answers.count(429)}" in out


def test_burst_video_ids_are_distinct_valid_youtube_ids() -> None:
    ids = video_ids(6)
    assert len(set(ids)) == 6
    assert [parse_video_id(f"https://www.youtube.com/watch?v={i}") for i in ids] == ids
