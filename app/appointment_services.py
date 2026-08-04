from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from app.models import (
    Appointment,
    AppointmentStatus,
    AvailabilitySlot,
    AvailabilityStatus,
    Customer,
    Service,
    Staff,
)
from app.schemas import AppointmentCreate


class AppointmentNotFoundError(Exception):
    """Raised when an appointment cannot be found."""


class InvalidAppointmentError(Exception):
    """Raised when appointment data violates a business rule."""


class AppointmentConflictError(Exception):
    """Raised when a requested slot or booking conflicts."""


def _generate_reference_number() -> str:
    """Generate a short unique-looking appointment reference."""

    return f"APT-{uuid4().hex[:8].upper()}"


def _get_active_customer(
    database: Session,
    customer_id: int,
) -> Customer:
    customer = database.get(Customer, customer_id)

    if customer is None:
        raise InvalidAppointmentError(
            f"Customer {customer_id} was not found."
        )

    if not customer.is_active:
        raise InvalidAppointmentError(
            "Inactive customers cannot create appointments."
        )

    return customer


def _get_active_service(
    database: Session,
    service_id: int,
) -> Service:
    service = database.get(Service, service_id)

    if service is None:
        raise InvalidAppointmentError(
            f"Service {service_id} was not found."
        )

    if not service.is_active:
        raise InvalidAppointmentError(
            "Inactive services cannot be booked."
        )

    return service


def _get_active_staff(
    database: Session,
    staff_id: int,
) -> Staff:
    statement = (
        select(Staff)
        .options(selectinload(Staff.services))
        .where(Staff.id == staff_id)
    )

    staff = database.scalar(statement)

    if staff is None:
        raise InvalidAppointmentError(
            f"Staff member {staff_id} was not found."
        )

    if not staff.is_active:
        raise InvalidAppointmentError(
            "Inactive staff members cannot receive appointments."
        )

    return staff


def _get_available_slot(
    database: Session,
    slot_id: int,
) -> AvailabilitySlot:
    slot = database.get(AvailabilitySlot, slot_id)

    if slot is None:
        raise InvalidAppointmentError(
            f"Availability slot {slot_id} was not found."
        )

    if slot.status != AvailabilityStatus.AVAILABLE:
        raise AppointmentConflictError(
            "The selected appointment slot is not available."
        )

    return slot


def get_appointment_by_id(
    database: Session,
    appointment_id: int,
) -> Appointment:
    appointment = database.get(
        Appointment,
        appointment_id,
    )

    if appointment is None:
        raise AppointmentNotFoundError(
            f"Appointment {appointment_id} was not found."
        )

    return appointment


def list_appointments(
    database: Session,
    customer_id: int | None = None,
    staff_id: int | None = None,
    service_id: int | None = None,
    appointment_status: AppointmentStatus | None = None,
) -> list[Appointment]:
    statement = select(Appointment).order_by(
        Appointment.start_datetime
    )

    if customer_id is not None:
        statement = statement.where(
            Appointment.customer_id == customer_id
        )

    if staff_id is not None:
        statement = statement.where(
            Appointment.staff_id == staff_id
        )

    if service_id is not None:
        statement = statement.where(
            Appointment.service_id == service_id
        )

    if appointment_status is not None:
        statement = statement.where(
            Appointment.status == appointment_status
        )

    return list(
        database.scalars(statement).all()
    )


def create_appointment(
    database: Session,
    appointment_data: AppointmentCreate,
) -> Appointment:
    """Validate and create a confirmed appointment."""

    _get_active_customer(
        database,
        appointment_data.customer_id,
    )

    _get_active_service(
        database,
        appointment_data.service_id,
    )

    staff = _get_active_staff(
        database,
        appointment_data.staff_id,
    )

    slot = _get_available_slot(
        database,
        appointment_data.slot_id,
    )

    staff_service_ids = {
        service.id
        for service in staff.services
    }

    if appointment_data.service_id not in staff_service_ids:
        raise InvalidAppointmentError(
            "The selected service is not assigned to this staff member."
        )

    if slot.staff_id != appointment_data.staff_id:
        raise InvalidAppointmentError(
            "The selected slot does not belong to this staff member."
        )

    if slot.service_id != appointment_data.service_id:
        raise InvalidAppointmentError(
            "The selected slot does not belong to this service."
        )

    existing_customer_booking = database.scalar(
        select(Appointment).where(
            Appointment.customer_id == appointment_data.customer_id,
            Appointment.start_datetime == slot.start_datetime,
            Appointment.status == AppointmentStatus.CONFIRMED,
        )
    )

    if existing_customer_booking is not None:
        raise AppointmentConflictError(
            "The customer already has an appointment at this time."
        )

    appointment = Appointment(
        reference_number=_generate_reference_number(),
        customer_id=appointment_data.customer_id,
        service_id=appointment_data.service_id,
        staff_id=appointment_data.staff_id,
        slot_id=slot.id,
        start_datetime=slot.start_datetime,
        end_datetime=slot.end_datetime,
        status=AppointmentStatus.CONFIRMED,
        customer_notes=appointment_data.customer_notes,
    )

    slot.status = AvailabilityStatus.BOOKED

    database.add(appointment)

    try:
        database.commit()
    except IntegrityError as error:
        database.rollback()

        raise AppointmentConflictError(
            "The selected slot was booked by another request."
        ) from error

    database.refresh(appointment)

    return appointment


def cancel_appointment(
    database: Session,
    appointment_id: int,
    cancellation_reason: str,
) -> Appointment:
    """Cancel an appointment and release its slot."""

    appointment = get_appointment_by_id(
        database,
        appointment_id,
    )

    if appointment.status != AppointmentStatus.CONFIRMED:
        raise InvalidAppointmentError(
            "Only confirmed appointments can be cancelled."
        )

    slot = database.get(
        AvailabilitySlot,
        appointment.slot_id,
    )

    if slot is not None:
        slot.status = AvailabilityStatus.AVAILABLE

    appointment.status = AppointmentStatus.CANCELLED_BY_CUSTOMER
    appointment.cancellation_reason = cancellation_reason
    appointment.updated_at = datetime.now()

    database.commit()
    database.refresh(appointment)

    return appointment


def reschedule_appointment(
    database: Session,
    appointment_id: int,
    new_slot_id: int,
) -> Appointment:
    """Move a confirmed appointment to another available slot."""

    appointment = get_appointment_by_id(
        database,
        appointment_id,
    )

    if appointment.status != AppointmentStatus.CONFIRMED:
        raise InvalidAppointmentError(
            "Only confirmed appointments can be rescheduled."
        )

    if appointment.slot_id == new_slot_id:
        raise InvalidAppointmentError(
            "The new slot must be different from the current slot."
        )

    new_slot = _get_available_slot(
        database,
        new_slot_id,
    )

    if new_slot.service_id != appointment.service_id:
        raise InvalidAppointmentError(
            "The new slot must provide the same service."
        )

    new_staff = _get_active_staff(
        database,
        new_slot.staff_id,
    )

    assigned_service_ids = {
        service.id
        for service in new_staff.services
    }

    if appointment.service_id not in assigned_service_ids:
        raise InvalidAppointmentError(
            "The new staff member does not provide this service."
        )

    old_slot = database.get(
        AvailabilitySlot,
        appointment.slot_id,
    )

    if old_slot is not None:
        old_slot.status = AvailabilityStatus.AVAILABLE

    new_slot.status = AvailabilityStatus.BOOKED

    appointment.staff_id = new_slot.staff_id
    appointment.slot_id = new_slot.id
    appointment.start_datetime = new_slot.start_datetime
    appointment.end_datetime = new_slot.end_datetime
    appointment.updated_at = datetime.now()

    try:
        database.commit()
    except IntegrityError as error:
        database.rollback()

        raise AppointmentConflictError(
            "The new slot was booked by another request."
        ) from error

    database.refresh(appointment)

    return appointment

def reschedule_appointment(
    database,
    appointment_id: int,
    new_slot_id: int,
):
    """Move an existing confirmed appointment to a new available slot."""

    appointment = (
        database.query(Appointment)
        .filter(Appointment.id == appointment_id)
        .first()
    )

    if appointment is None:
        raise InvalidAppointmentError("appointment was not found.")

    old_slot = (
        database.query(AvailabilitySlot)
        .filter(AvailabilitySlot.id == appointment.slot_id)
        .first()
    )

    new_slot = (
        database.query(AvailabilitySlot)
        .filter(AvailabilitySlot.id == new_slot_id)
        .first()
    )

    if new_slot is None:
        raise InvalidAppointmentError("new slot was not found.")

    if new_slot.status != AvailabilityStatus.AVAILABLE:
        raise AppointmentConflictError("the new slot is not available.")

    if old_slot is not None:
        old_slot.status = AvailabilityStatus.AVAILABLE

    new_slot.status = AvailabilityStatus.BOOKED

    appointment.slot_id = new_slot.id
    appointment.staff_id = new_slot.staff_id
    appointment.service_id = new_slot.service_id
    appointment.start_datetime = new_slot.start_datetime
    appointment.end_datetime = new_slot.end_datetime

    database.commit()
    database.refresh(appointment)

    return appointment