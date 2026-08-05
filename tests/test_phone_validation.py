from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from langchain_core.messages import HumanMessage

from app.ai import agent
from app.ai.agent import (
    determine_next_question,
    extract_details,
    get_or_create_customer_from_details,
    normalize_sri_lankan_phone_number,
    resolve_customer_details,
)
from app.database import Base
from app.models import Customer


def test_normalizes_local_sri_lankan_mobile() -> None:
    assert normalize_sri_lankan_phone_number("0771234567") == (
        "+94771234567"
    )


def test_preserves_normalized_sri_lankan_mobile() -> None:
    assert normalize_sri_lankan_phone_number("+94771234567") == (
        "+94771234567"
    )


def test_normalizes_94_sri_lankan_mobile() -> None:
    assert normalize_sri_lankan_phone_number("94771234567") == (
        "+94771234567"
    )


def test_normalizes_nine_digit_sri_lankan_mobile() -> None:
    assert normalize_sri_lankan_phone_number("771234567") == (
        "+94771234567"
    )


def test_rejects_too_long_sri_lankan_mobile() -> None:
    assert normalize_sri_lankan_phone_number("07756688791") is None


def test_invalid_phone_keeps_name_and_asks_for_valid_phone() -> None:
    state = {
        "messages": [
            HumanMessage(
                content="My name is Kivaane, 07756688791",
            ),
        ],
        "intent": "book_appointment",
        "slot_id": 7,
        "selected_slot_summary": "Dental care at 10:00 AM",
        "available_services": [{"id": 2, "name": "Dental care"}],
    }

    updates = extract_details(state)

    assert updates["customer_name"] == "Kivaane"
    assert updates["customer_phone_number"] is None
    assert updates["customer_phone_invalid"] is True

    response = determine_next_question(
        {
            **state,
            **updates,
        }
    )["next_question"]

    assert response == (
        "Please enter a valid Sri Lankan phone number, for example "
        "0771234567 or +94771234567."
    )


def test_name_first_then_valid_phone_continues_booking(monkeypatch) -> None:
    name_updates = extract_details(
        {
            "messages": [
                HumanMessage(content="My name is Kivaane Anton"),
            ],
            "intent": "book_appointment",
            "slot_id": 7,
        }
    )

    assert name_updates["customer_name"] == "Kivaane Anton"

    phone_state = {
        "messages": [
            HumanMessage(content="0771234567"),
        ],
        "intent": "book_appointment",
        "slot_id": 7,
        **name_updates,
    }
    phone_updates = extract_details(phone_state)

    monkeypatch.setattr(
        agent,
        "get_or_create_customer_from_details",
        lambda full_name, phone_number: {
            "id": 11,
            "full_name": full_name,
            "phone_number": phone_number,
        },
    )

    resolved = resolve_customer_details(
        {
            **phone_state,
            **phone_updates,
        }
    )

    assert resolved["customer_id"] == 11
    assert resolved["customer_phone_number"] == "+94771234567"


def test_existing_canonical_customer_is_reused_for_local_number(
    monkeypatch,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    test_session_local = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)

    database = test_session_local()
    existing_customer = Customer(
        full_name="Existing Name",
        phone_number="+94771234567",
        is_active=True,
    )
    database.add(existing_customer)
    database.commit()
    existing_id = existing_customer.id
    database.close()

    monkeypatch.setattr(agent, "SessionLocal", test_session_local)

    result = get_or_create_customer_from_details(
        full_name="Kivaane Anton",
        phone_number="0771234567",
    )

    database = test_session_local()

    try:
        assert result["id"] == existing_id
        assert result["phone_number"] == "+94771234567"
        assert database.query(Customer).count() == 1
    finally:
        database.close()
