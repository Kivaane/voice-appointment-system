from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from app.config import get_settings


def get_business_timezone() -> ZoneInfo:
    """Return the configured timezone used for appointment wall time."""

    return ZoneInfo(get_settings().business_timezone)


def utc_now() -> datetime:
    """Return an aware UTC timestamp for persistence metadata."""

    return datetime.now(timezone.utc)


def business_now(now: datetime | None = None) -> datetime:
    """Return an aware current datetime in the business timezone."""

    reference = now or utc_now()

    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)

    return reference.astimezone(get_business_timezone())


def business_today(now: datetime | None = None) -> date:
    """Return today's date at the business, independent of server TZ."""

    return business_now(now).date()


def business_now_naive(now: datetime | None = None) -> datetime:
    """Return business wall time for legacy naive schedule columns."""

    return business_now(now).replace(tzinfo=None)


def to_business_datetime(value: datetime) -> datetime:
    """Interpret legacy naive values as business wall time."""

    if value.tzinfo is None:
        return value.replace(tzinfo=get_business_timezone())

    return value.astimezone(get_business_timezone())
