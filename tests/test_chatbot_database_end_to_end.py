from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai import agent
from app.database import Base
from app.models import (
    Appointment,
    AppointmentStatus,
    AvailabilitySlot,
    AvailabilityStatus,
    Customer,
    Service,
    Staff,
)


def build_action_database():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    with session_local() as database:
        service = Service(
            name="Dental care",
            description="Dental care",
            duration_minutes=30,
            price=Decimal("3500.00"),
            is_active=True,
        )
        staff = Staff(
            full_name="Dr. Perera",
            email="e2e.perera@example.com",
            is_active=True,
        )
        staff.services.append(service)
        customer = Customer(
            full_name="Kivaane Anton",
            phone_number="0774588691",
            is_active=True,
        )
        database.add_all([service, staff, customer])
        database.flush()

        starts = [
            datetime(2026, 8, 6, 9, 0),
            datetime(2026, 8, 7, 10, 0),
        ]
        slots = [
            AvailabilitySlot(
                staff_id=staff.id,
                service_id=service.id,
                start_datetime=start,
                end_datetime=start + timedelta(minutes=30),
                status=AvailabilityStatus.AVAILABLE,
            )
            for start in starts
        ]
        database.add_all(slots)
        database.commit()

        identifiers = {
            "customer_id": customer.id,
            "service_id": service.id,
            "staff_id": staff.id,
            "first_slot_id": slots[0].id,
            "second_slot_id": slots[1].id,
        }

    return session_local, identifiers


def test_confirmed_chatbot_actions_match_database_outcomes(monkeypatch) -> None:
    session_local, ids = build_action_database()
    monkeypatch.setattr(agent, "SessionLocal", session_local)

    booking_state = {
        "intent": "book_appointment",
        "confirmation_status": "confirmed",
        "booking_summary": "Confirm dental booking",
        "customer_id": ids["customer_id"],
        "customer_name": "Kivaane Anton",
        "service_id": ids["service_id"],
        "service_name": "Dental care",
        "staff_id": ids["staff_id"],
        "staff_name": "Dr. Perera",
        "slot_id": ids["first_slot_id"],
        "requested_date": "2026-08-06",
    }
    booked = agent.confirm_or_reject_booking(booking_state)

    with session_local() as database:
        appointment = database.scalar(select(Appointment))
        first_slot = database.get(
            AvailabilitySlot,
            ids["first_slot_id"],
        )
        assert database.scalar(select(func.count(Appointment.id))) == 1
        assert appointment.status == AppointmentStatus.CONFIRMED
        assert first_slot.status == AvailabilityStatus.BOOKED
        appointment_id = appointment.id

    duplicate = agent.confirm_or_reject_booking(
        {**booking_state, **booked, "booking_summary": None}
    )
    assert duplicate == {}
    with session_local() as database:
        assert database.scalar(select(func.count(Appointment.id))) == 1

    rescheduled = agent.confirm_or_reject_booking(
        {
            "intent": "reschedule_appointment",
            "confirmation_status": "confirmed",
            "booking_summary": "Confirm reschedule",
            "appointment_id": appointment_id,
            "current_slot_id": ids["first_slot_id"],
            "service_id": ids["service_id"],
            "service_name": "Dental care",
            "staff_id": ids["staff_id"],
            "staff_name": "Dr. Perera",
            "slot_id": ids["second_slot_id"],
            "requested_date": "2026-08-07",
        }
    )
    assert "rescheduled" in rescheduled["next_question"].lower()

    with session_local() as database:
        appointment = database.get(Appointment, appointment_id)
        first_slot = database.get(
            AvailabilitySlot,
            ids["first_slot_id"],
        )
        second_slot = database.get(
            AvailabilitySlot,
            ids["second_slot_id"],
        )
        assert appointment.slot_id == ids["second_slot_id"]
        assert first_slot.status == AvailabilityStatus.AVAILABLE
        assert second_slot.status == AvailabilityStatus.BOOKED

    cancelled = agent.confirm_or_reject_booking(
        {
            "intent": "cancel_appointment",
            "confirmation_status": "confirmed",
            "booking_summary": "Confirm cancellation",
            "appointment_id": appointment_id,
            "appointment_reference_number": "APT-E2E",
            "cancellation_reason": "Customer requested cancellation",
        }
    )
    assert "cancelled" in cancelled["next_question"].lower()

    with session_local() as database:
        appointment = database.get(Appointment, appointment_id)
        second_slot = database.get(
            AvailabilitySlot,
            ids["second_slot_id"],
        )
        assert appointment.status == AppointmentStatus.CANCELLED_BY_CUSTOMER
        assert appointment.cancellation_reason == (
            "Customer requested cancellation"
        )
        assert second_slot.status == AvailabilityStatus.AVAILABLE
        assert database.scalar(select(func.count(Appointment.id))) == 1
