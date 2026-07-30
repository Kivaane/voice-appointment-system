from datetime import datetime
from types import SimpleNamespace

from app.ai import tools
from app.models import AppointmentStatus


class FakeDatabase:
    """Minimal fake database session used by tool tests."""

    closed = False

    def close(self) -> None:
        self.closed = True


def test_book_appointment_tool(
    monkeypatch,
) -> None:
    database = FakeDatabase()

    fake_appointment = SimpleNamespace(
        id=1,
        reference_number="APT-TEST001",
        customer_id=10,
        service_id=20,
        staff_id=30,
        slot_id=40,
        start_datetime=datetime(
            2026,
            8,
            1,
            10,
            0,
        ),
        end_datetime=datetime(
            2026,
            8,
            1,
            10,
            30,
        ),
        status=AppointmentStatus.CONFIRMED,
    )

    def fake_create_appointment(
        database,
        appointment_data,
    ):
        assert appointment_data.customer_id == 10
        assert appointment_data.service_id == 20
        assert appointment_data.staff_id == 30
        assert appointment_data.slot_id == 40
        assert appointment_data.customer_notes == "First visit"

        return fake_appointment

    monkeypatch.setattr(
        tools,
        "SessionLocal",
        lambda: database,
    )

    monkeypatch.setattr(
        tools,
        "create_appointment",
        fake_create_appointment,
    )

    result = tools.book_appointment.invoke(
        {
            "customer_id": 10,
            "service_id": 20,
            "staff_id": 30,
            "slot_id": 40,
            "customer_notes": "First visit",
        }
    )

    assert result == {
        "appointment_id": 1,
        "reference_number": "APT-TEST001",
        "customer_id": 10,
        "service_id": 20,
        "staff_id": 30,
        "slot_id": 40,
        "start_datetime": "2026-08-01T10:00:00",
        "end_datetime": "2026-08-01T10:30:00",
        "status": "CONFIRMED",
    }

    assert database.closed is True


def test_cancel_appointment_tool(
    monkeypatch,
) -> None:
    database = FakeDatabase()

    fake_appointment = SimpleNamespace(
        id=1,
        reference_number="APT-TEST001",
        slot_id=40,
        status=AppointmentStatus.CANCELLED_BY_CUSTOMER,
        cancellation_reason="Customer request",
    )

    def fake_cancel_appointment(
        database,
        appointment_id,
        cancellation_reason,
    ):
        assert appointment_id == 1
        assert cancellation_reason == "Customer request"

        return fake_appointment

    monkeypatch.setattr(
        tools,
        "SessionLocal",
        lambda: database,
    )

    monkeypatch.setattr(
        tools,
        "cancel_appointment",
        fake_cancel_appointment,
    )

    result = tools.cancel_existing_appointment.invoke(
        {
            "appointment_id": 1,
            "cancellation_reason": "Customer request",
        }
    )

    assert result == {
        "appointment_id": 1,
        "reference_number": "APT-TEST001",
        "slot_id": 40,
        "status": "CANCELLED_BY_CUSTOMER",
        "cancellation_reason": "Customer request",
    }

    assert database.closed is True


def test_reschedule_appointment_tool(
    monkeypatch,
) -> None:
    database = FakeDatabase()

    fake_appointment = SimpleNamespace(
        id=1,
        reference_number="APT-TEST001",
        service_id=20,
        staff_id=31,
        slot_id=41,
        start_datetime=datetime(
            2026,
            8,
            2,
            14,
            0,
        ),
        end_datetime=datetime(
            2026,
            8,
            2,
            14,
            30,
        ),
        status=AppointmentStatus.CONFIRMED,
    )

    def fake_reschedule_appointment(
        database,
        appointment_id,
        new_slot_id,
    ):
        assert appointment_id == 1
        assert new_slot_id == 41

        return fake_appointment

    monkeypatch.setattr(
        tools,
        "SessionLocal",
        lambda: database,
    )

    monkeypatch.setattr(
        tools,
        "reschedule_appointment",
        fake_reschedule_appointment,
    )

    result = tools.reschedule_existing_appointment.invoke(
        {
            "appointment_id": 1,
            "new_slot_id": 41,
        }
    )

    assert result == {
        "appointment_id": 1,
        "reference_number": "APT-TEST001",
        "service_id": 20,
        "staff_id": 31,
        "slot_id": 41,
        "start_datetime": "2026-08-02T14:00:00",
        "end_datetime": "2026-08-02T14:30:00",
        "status": "CONFIRMED",
    }

    assert database.closed is True