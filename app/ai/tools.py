from datetime import date

from langchain.tools import tool

from app.availability_services import list_availability_slots
from app.config import get_settings
from app.database import SessionLocal
from app.models import AvailabilityStatus
from app.services import list_services
from app.appointment_services import (
    cancel_appointment,
    create_appointment,
    reschedule_appointment,
)
from app.schemas import AppointmentCreate

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

@tool
def book_appointment(
    customer_id: int,
    service_id: int,
    staff_id: int,
    slot_id: int,
    customer_notes: str | None = None,
) -> dict[str, object]:
    """Create a confirmed appointment using an available slot.

    Use this tool only after the customer has confirmed the complete
    booking details and human approval has been received.
    """

    database = SessionLocal()

    try:
        appointment = create_appointment(
            database=database,
            appointment_data=AppointmentCreate(
                customer_id=customer_id,
                service_id=service_id,
                staff_id=staff_id,
                slot_id=slot_id,
                customer_notes=customer_notes,
            ),
        )

        return {
            "appointment_id": appointment.id,
            "reference_number": appointment.reference_number,
            "customer_id": appointment.customer_id,
            "service_id": appointment.service_id,
            "staff_id": appointment.staff_id,
            "slot_id": appointment.slot_id,
            "start_datetime": (
                appointment.start_datetime.isoformat()
            ),
            "end_datetime": (
                appointment.end_datetime.isoformat()
            ),
            "status": appointment.status.value,
        }

    finally:
        database.close()


@tool
def cancel_existing_appointment(
    appointment_id: int,
    cancellation_reason: str,
) -> dict[str, object]:
    """Cancel a confirmed appointment and release its slot.

    Use this tool only after the customer confirms that the appointment
    should be cancelled.
    """

    database = SessionLocal()

    try:
        appointment = cancel_appointment(
            database=database,
            appointment_id=appointment_id,
            cancellation_reason=cancellation_reason,
        )

        return {
            "appointment_id": appointment.id,
            "reference_number": appointment.reference_number,
            "slot_id": appointment.slot_id,
            "status": appointment.status.value,
            "cancellation_reason": (
                appointment.cancellation_reason
            ),
        }

    finally:
        database.close()


@tool
def reschedule_existing_appointment(
    appointment_id: int,
    new_slot_id: int,
) -> dict[str, object]:
    """Move a confirmed appointment to another available slot.

    Use this tool only after availability has been checked and the
    customer confirms the new appointment slot.
    """

    database = SessionLocal()

    try:
        appointment = reschedule_appointment(
            database=database,
            appointment_id=appointment_id,
            new_slot_id=new_slot_id,
        )

        return {
            "appointment_id": appointment.id,
            "reference_number": appointment.reference_number,
            "service_id": appointment.service_id,
            "staff_id": appointment.staff_id,
            "slot_id": appointment.slot_id,
            "start_datetime": (
                appointment.start_datetime.isoformat()
            ),
            "end_datetime": (
                appointment.end_datetime.isoformat()
            ),
            "status": appointment.status.value,
        }

    finally:
        database.close()