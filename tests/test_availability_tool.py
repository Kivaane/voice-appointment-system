from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.ai import tools
from app.database import Base
from app.models import (
    AvailabilitySlot,
    AvailabilityStatus,
    Service,
    Staff,
)


def test_check_available_slots_returns_only_matching_slots(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
    )

    Base.metadata.create_all(bind=engine)

    test_session_local = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    with test_session_local() as database:
        service = Service(
            name="Dental care",
            description="Dental appointment",
            duration_minutes=30,
            is_active=True,
        )

        staff = Staff(
            full_name="Dr. Perera",
            email="perera@example.com",
            is_active=True,
        )

        database.add_all(
            [
                service,
                staff,
            ]
        )
        database.commit()
        database.refresh(service)
        database.refresh(staff)

        database.add_all(
            [
                AvailabilitySlot(
                    staff_id=staff.id,
                    service_id=service.id,
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
                    status=AvailabilityStatus.AVAILABLE,
                ),
                AvailabilitySlot(
                    staff_id=staff.id,
                    service_id=service.id,
                    start_datetime=datetime(
                        2026,
                        8,
                        1,
                        11,
                        0,
                    ),
                    end_datetime=datetime(
                        2026,
                        8,
                        1,
                        11,
                        30,
                    ),
                    status=AvailabilityStatus.BLOCKED,
                ),
            ]
        )

        database.commit()

        service_id = service.id
        staff_id = staff.id

    monkeypatch.setattr(
        tools,
        "SessionLocal",
        test_session_local,
    )

    result = tools.check_available_slots.invoke(
        {
            "service_id": service_id,
            "requested_date": "2026-08-01",
            "staff_id": staff_id,
        }
    )

    assert len(result) == 1

    assert result[0]["service_id"] == service_id
    assert result[0]["staff_id"] == staff_id
    assert result[0]["status"] == "AVAILABLE"

    assert result[0]["start_datetime"] == (
        "2026-08-01T10:00:00"
    )


def test_check_available_slots_rejects_invalid_date() -> None:
    try:
        tools.check_available_slots.invoke(
            {
                "service_id": 1,
                "requested_date": "01-08-2026",
            }
        )
    except ValueError as error:
        assert str(error) == (
            "requested_date must use the format YYYY-MM-DD."
        )
    else:
        raise AssertionError(
            "Expected an invalid date to raise ValueError."
        )