"""Model health helpers (D36). The Redis store itself is tested against real Redis in
tests/integration/test_model_health.py."""

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

from worker.llm.health import seconds_until_midnight

PACIFIC_STANDARD = timezone(timedelta(hours=-8))


def test_a_spent_daily_quota_is_benched_until_the_next_midnight() -> None:
    now = datetime(2026, 11, 10, 23, 0, tzinfo=PACIFIC_STANDARD)
    assert seconds_until_midnight(PACIFIC_STANDARD, now) == 3600


def test_the_reset_is_found_from_a_time_in_any_zone() -> None:
    now = datetime(2026, 11, 10, 7, 30, tzinfo=UTC)  # 23:30 the evening before, Pacific
    assert seconds_until_midnight(PACIFIC_STANDARD, now) == 1800


def test_at_midnight_the_next_reset_is_a_day_away() -> None:
    now = datetime(2026, 11, 10, 0, 0, tzinfo=PACIFIC_STANDARD)
    assert seconds_until_midnight(PACIFIC_STANDARD, now) == 24 * 3600


def test_a_day_with_a_clock_change_is_counted_in_real_seconds() -> None:
    try:
        pacific = ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError:  # Windows without the tzdata package; the worker image has it
        pytest.skip("no time zone database")
    # 1 Nov 2026: clocks go back at 02:00, so 00:30 -> next midnight is 24.5 real hours.
    now = datetime(2026, 11, 1, 0, 30, tzinfo=pacific)
    assert seconds_until_midnight(pacific, now) == 24.5 * 3600
