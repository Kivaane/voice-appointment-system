from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from app.ai.agent import format_time, parse_requested_date
from app.time_utils import business_now_naive, business_today


def test_tomorrow_uses_asia_colombo_business_date() -> None:
    near_midnight_utc = datetime(
        2026,
        8,
        4,
        18,
        45,
        tzinfo=timezone.utc,
    )

    with patch(
        "app.ai.agent.business_today",
        return_value=business_today(near_midnight_utc),
    ):
        assert parse_requested_date("tomorrow") == "2026-08-06"


def test_business_date_does_not_depend_on_server_timezone() -> None:
    reference_utc = datetime(
        2026,
        8,
        4,
        18,
        45,
        tzinfo=timezone.utc,
    )

    assert business_today(reference_utc).isoformat() == "2026-08-05"
    assert business_now_naive(reference_utc) == datetime(2026, 8, 5, 0, 15)


def test_aware_utc_slot_displays_in_business_timezone() -> None:
    assert format_time("2026-08-05T04:30:00+00:00") == "10:00 AM"


def test_business_timezone_is_configurable() -> None:
    with patch(
        "app.time_utils.get_settings",
        return_value=SimpleNamespace(business_timezone="UTC"),
    ):
        reference = datetime(2026, 8, 4, 23, 30, tzinfo=timezone.utc)
        assert business_today(reference).isoformat() == "2026-08-04"
