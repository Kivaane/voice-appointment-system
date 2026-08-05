from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from app.appointment_services import (
    AppointmentConflictError,
    create_appointment,
    reschedule_appointment,
)
from app.models import (
    Appointment,
    AvailabilitySlot,
    AvailabilityStatus,
    Customer,
    Service,
    Staff,
)
from app.schemas import AppointmentCreate
from conftest import TestingSessionLocal


def seed_booking_data():
    database = TestingSessionLocal()
    service = Service(
        name="Concurrency Dental",
        description="Test",
        duration_minutes=30,
        price=Decimal("1000"),
        is_active=True,
    )
    staff = Staff(
        full_name="Dr. Lock",
        email="lock@example.com",
        is_active=True,
    )
    staff.services.append(service)
    first_customer = Customer(
        full_name="First Customer",
        phone_number="+94770000001",
        is_active=True,
    )
    second_customer = Customer(
        full_name="Second Customer",
        phone_number="+94770000002",
        is_active=True,
    )
    start = datetime(2026, 8, 10, 9, 0)
    first_slot = AvailabilitySlot(
        staff=staff,
        service=service,
        start_datetime=start,
        end_datetime=start + timedelta(minutes=30),
        status=AvailabilityStatus.AVAILABLE,
    )
    second_slot = AvailabilitySlot(
        staff=staff,
        service=service,
        start_datetime=start + timedelta(hours=1),
        end_datetime=start + timedelta(hours=1, minutes=30),
        status=AvailabilityStatus.AVAILABLE,
    )
    database.add_all(
        [service, staff, first_customer, second_customer, first_slot, second_slot]
    )
    database.commit()
    ids = {
        "service": service.id,
        "staff": staff.id,
        "first_customer": first_customer.id,
        "second_customer": second_customer.id,
        "first_slot": first_slot.id,
        "second_slot": second_slot.id,
    }
    database.close()
    return ids


def appointment_data(ids, customer_key, slot_key):
    return AppointmentCreate(
        customer_id=ids[customer_key],
        service_id=ids["service"],
        staff_id=ids["staff"],
        slot_id=ids[slot_key],
    )


def test_two_attempts_for_one_slot_create_only_one_appointment() -> None:
    ids = seed_booking_data()

    with TestingSessionLocal() as first_database:
        create_appointment(
            first_database,
            appointment_data(ids, "first_customer", "first_slot"),
        )

    with TestingSessionLocal() as second_database:
        with pytest.raises(AppointmentConflictError):
            create_appointment(
                second_database,
                appointment_data(ids, "second_customer", "first_slot"),
            )

    with TestingSessionLocal() as database:
        count = database.scalar(select(func.count(Appointment.id)))
        slot = database.get(AvailabilitySlot, ids["first_slot"])
        assert count == 1
        assert slot is not None
        assert slot.status == AvailabilityStatus.BOOKED


def test_failed_reschedule_keeps_original_slot_and_appointment() -> None:
    ids = seed_booking_data()

    with TestingSessionLocal() as database:
        original = create_appointment(
            database,
            appointment_data(ids, "first_customer", "first_slot"),
        )
        create_appointment(
            database,
            appointment_data(ids, "second_customer", "second_slot"),
        )
        appointment_id = original.id

    with TestingSessionLocal() as database:
        with pytest.raises(AppointmentConflictError):
            reschedule_appointment(
                database,
                appointment_id,
                ids["second_slot"],
            )

    with TestingSessionLocal() as database:
        appointment = database.get(Appointment, appointment_id)
        old_slot = database.get(AvailabilitySlot, ids["first_slot"])
        assert appointment is not None
        assert appointment.slot_id == ids["first_slot"]
        assert old_slot is not None
        assert old_slot.status == AvailabilityStatus.BOOKED
