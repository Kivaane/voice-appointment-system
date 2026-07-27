from datetime import date, datetime, time, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import (
    AvailabilitySlot,
    AvailabilityStatus,
    Service,
    Staff,
)
from app.schemas import (
    AvailabilitySlotCreate,
    AvailabilitySlotUpdate,
)


class AvailabilitySlotNotFoundError(Exception):
    """Raised when an availability slot cannot be found."""


class InvalidAvailabilitySlotError(Exception):
    """Raised when availability information violates a business rule."""


class AvailabilityConflictError(Exception):
    """Raised when a staff member already has an overlapping slot."""


def get_availability_slot_by_id(
    database: Session,
    slot_id: int,
) -> AvailabilitySlot:
    """Return one availability slot by ID."""

    slot = database.get(AvailabilitySlot, slot_id)

    if slot is None:
        raise AvailabilitySlotNotFoundError(
            f"Availability slot {slot_id} was not found."
        )

    return slot


def _validate_time_range(
    start_datetime: datetime,
    end_datetime: datetime,
) -> None:
    """Ensure that the slot has a valid time range."""

    if end_datetime <= start_datetime:
        raise InvalidAvailabilitySlotError(
            "The end time must be later than the start time."
        )


def _get_active_staff(
    database: Session,
    staff_id: int,
) -> Staff:
    """Return an active staff member with assigned services."""

    statement = (
        select(Staff)
        .options(selectinload(Staff.services))
        .where(Staff.id == staff_id)
    )

    staff = database.scalar(statement)

    if staff is None:
        raise InvalidAvailabilitySlotError(
            f"Staff member {staff_id} was not found."
        )

    if not staff.is_active:
        raise InvalidAvailabilitySlotError(
            "Availability cannot be created for inactive staff."
        )

    return staff


def _get_active_service(
    database: Session,
    service_id: int,
) -> Service:
    """Return an active bookable service."""

    service = database.get(Service, service_id)

    if service is None:
        raise InvalidAvailabilitySlotError(
            f"Service {service_id} was not found."
        )

    if not service.is_active:
        raise InvalidAvailabilitySlotError(
            "Availability cannot be created for an inactive service."
        )

    return service


def _check_staff_service_assignment(
    staff: Staff,
    service_id: int,
) -> None:
    """Ensure that the staff member provides the requested service."""

    assigned_service_ids = {
        service.id
        for service in staff.services
    }

    if service_id not in assigned_service_ids:
        raise InvalidAvailabilitySlotError(
            "The selected service is not assigned to this staff member."
        )


def _check_for_overlap(
    database: Session,
    staff_id: int,
    start_datetime: datetime,
    end_datetime: datetime,
    excluded_slot_id: int | None = None,
) -> None:
    """Prevent overlapping slots for the same staff member."""

    statement = select(AvailabilitySlot).where(
        AvailabilitySlot.staff_id == staff_id,
        AvailabilitySlot.start_datetime < end_datetime,
        AvailabilitySlot.end_datetime > start_datetime,
    )

    if excluded_slot_id is not None:
        statement = statement.where(
            AvailabilitySlot.id != excluded_slot_id
        )

    overlapping_slot = database.scalar(statement)

    if overlapping_slot is not None:
        raise AvailabilityConflictError(
            "This time overlaps with another slot for the staff member."
        )


def create_availability_slot(
    database: Session,
    slot_data: AvailabilitySlotCreate,
) -> AvailabilitySlot:
    """Create a new availability slot."""

    _validate_time_range(
        slot_data.start_datetime,
        slot_data.end_datetime,
    )

    staff = _get_active_staff(
        database,
        slot_data.staff_id,
    )

    _get_active_service(
        database,
        slot_data.service_id,
    )

    _check_staff_service_assignment(
        staff,
        slot_data.service_id,
    )

    _check_for_overlap(
        database=database,
        staff_id=slot_data.staff_id,
        start_datetime=slot_data.start_datetime,
        end_datetime=slot_data.end_datetime,
    )

    slot = AvailabilitySlot(
        staff_id=slot_data.staff_id,
        service_id=slot_data.service_id,
        start_datetime=slot_data.start_datetime,
        end_datetime=slot_data.end_datetime,
        status=AvailabilityStatus.AVAILABLE,
    )

    database.add(slot)
    database.commit()
    database.refresh(slot)

    return slot


def list_availability_slots(
    database: Session,
    staff_id: int | None = None,
    service_id: int | None = None,
    requested_date: date | None = None,
    slot_status: AvailabilityStatus | None = None,
) -> list[AvailabilitySlot]:
    """Return availability slots using optional filters."""

    statement = select(AvailabilitySlot).order_by(
        AvailabilitySlot.start_datetime
    )

    if staff_id is not None:
        statement = statement.where(
            AvailabilitySlot.staff_id == staff_id
        )

    if service_id is not None:
        statement = statement.where(
            AvailabilitySlot.service_id == service_id
        )

    if slot_status is not None:
        statement = statement.where(
            AvailabilitySlot.status == slot_status
        )

    if requested_date is not None:
        day_start = datetime.combine(
            requested_date,
            time.min,
        )

        next_day_start = day_start + timedelta(days=1)

        statement = statement.where(
            AvailabilitySlot.start_datetime >= day_start,
            AvailabilitySlot.start_datetime < next_day_start,
        )

    return list(
        database.scalars(statement).all()
    )


def update_availability_slot(
    database: Session,
    slot_id: int,
    slot_data: AvailabilitySlotUpdate,
) -> AvailabilitySlot:
    """Update an existing availability slot."""

    slot = get_availability_slot_by_id(
        database,
        slot_id,
    )

    if slot.status == AvailabilityStatus.BOOKED:
        raise InvalidAvailabilitySlotError(
            "A booked slot cannot be edited directly."
        )

    update_values = slot_data.model_dump(
        exclude_unset=True
    )

    new_start = update_values.get(
        "start_datetime",
        slot.start_datetime,
    )

    new_end = update_values.get(
        "end_datetime",
        slot.end_datetime,
    )

    _validate_time_range(
        new_start,
        new_end,
    )

    if (
        "start_datetime" in update_values
        or "end_datetime" in update_values
    ):
        _check_for_overlap(
            database=database,
            staff_id=slot.staff_id,
            start_datetime=new_start,
            end_datetime=new_end,
            excluded_slot_id=slot.id,
        )

    for field_name, field_value in update_values.items():
        setattr(slot, field_name, field_value)

    database.commit()
    database.refresh(slot)

    return slot


def block_availability_slot(
    database: Session,
    slot_id: int,
) -> AvailabilitySlot:
    """Block an unbooked availability slot."""

    slot = get_availability_slot_by_id(
        database,
        slot_id,
    )

    if slot.status == AvailabilityStatus.BOOKED:
        raise InvalidAvailabilitySlotError(
            "A booked slot cannot be blocked directly."
        )

    slot.status = AvailabilityStatus.BLOCKED

    database.commit()
    database.refresh(slot)

    return slot


def activate_availability_slot(
    database: Session,
    slot_id: int,
) -> AvailabilitySlot:
    """Make a blocked unbooked slot available again."""

    slot = get_availability_slot_by_id(
        database,
        slot_id,
    )

    if slot.status == AvailabilityStatus.BOOKED:
        raise InvalidAvailabilitySlotError(
            "A booked slot cannot be activated directly."
        )

    slot.status = AvailabilityStatus.AVAILABLE

    database.commit()
    database.refresh(slot)

    return slot