from datetime import date

from langchain.tools import tool

from app.availability_services import list_availability_slots
from app.config import get_settings
from app.database import SessionLocal
from app.models import AvailabilityStatus
from app.services import list_services


@tool
def list_available_services() -> list[dict[str, object]]:
    """Return all active appointment services offered by the business.

    Use this tool when a customer asks what services are available,
    what they can book, or which appointment types are offered.
    """

    database = SessionLocal()
    settings = get_settings()

    try:
        services = list_services(
            database=database,
            include_inactive=False,
        )

        return [
            {
                "id": service.id,
                "name": service.name,
                "description": service.description,
                "duration_minutes": service.duration_minutes,
                "price": (
                    float(service.price)
                    if service.price is not None
                    else None
                ),
                "currency": settings.currency,
            }
            for service in services
        ]

    finally:
        database.close()


@tool
def check_available_slots(
    service_id: int,
    requested_date: str,
    staff_id: int | None = None,
) -> list[dict[str, object]]:
    """Return available appointment slots for a service and date.

    The requested date must use the ISO format YYYY-MM-DD.
    Optionally provide a staff ID to filter results.
    """

    try:
        parsed_date = date.fromisoformat(requested_date)
    except ValueError as error:
        raise ValueError(
            "requested_date must use the format YYYY-MM-DD."
        ) from error

    database = SessionLocal()

    try:
        slots = list_availability_slots(
            database=database,
            service_id=service_id,
            staff_id=staff_id,
            requested_date=parsed_date,
            slot_status=AvailabilityStatus.AVAILABLE,
        )

        return [
            {
                "slot_id": slot.id,
                "service_id": slot.service_id,
                "staff_id": slot.staff_id,
                "start_datetime": slot.start_datetime.isoformat(),
                "end_datetime": slot.end_datetime.isoformat(),
                "status": slot.status.value,
            }
            for slot in slots
        ]

    finally:
        database.close()