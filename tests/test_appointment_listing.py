from unittest.mock import patch

from langchain_core.messages import HumanMessage

from app.ai.agent import (
    call_model,
    detect_intent,
    determine_next_question,
)


APPOINTMENT = {
    "appointment_id": 12,
    "appointment_reference_number": "APT-871E6728",
    "appointment_status": "CONFIRMED",
    "customer_id": 11,
    "customer_phone_number": "+94771234567",
    "service_id": 2,
    "service_name": "Dental care",
    "staff_id": 5,
    "staff_name": "Dr. Perera",
    "current_slot_id": 7,
    "start_datetime": "2026-08-05T10:00:00",
    "end_datetime": "2026-08-05T10:30:00",
}

SERVICES = [
    {
        "id": 2,
        "name": "Dental care",
        "description": "Dental checkups.",
        "duration_minutes": 30,
        "price": 3500,
    }
]


def test_lists_customer_upcoming_appointments_without_model() -> None:
    state = {
        "messages": [
            HumanMessage(content="can u tell me all my appointments"),
        ],
        "intent": "book_appointment",
        "customer_id": 11,
        "available_services": SERVICES,
    }
    intent_updates = detect_intent(state)

    assert intent_updates["intent"] == "view_appointments"

    with patch(
        "app.ai.agent.get_upcoming_appointments_for_customer",
        return_value=[APPOINTMENT],
    ) as mocked_lookup:
        question_updates = determine_next_question(
            {
                **state,
                **intent_updates,
            }
        )

    mocked_lookup.assert_called_once_with(
        customer_id=11,
        phone_number=None,
    )
    response = str(question_updates["next_question"])

    assert "Here are your upcoming appointments:" in response
    assert "Reference: APT-871E6728" in response
    assert "Service: Dental care" in response
    assert "Doctor/Staff: Dr. Perera" in response
    assert "Status: CONFIRMED" in response

    with patch("app.ai.agent.get_chat_model") as mocked_model:
        call_model(
            {
                **state,
                **intent_updates,
                **question_updates,
            }
        )

    mocked_model.assert_not_called()


def test_appointment_status_by_reference() -> None:
    state = {
        "messages": [
            HumanMessage(content="appointment status APT-871E6728"),
        ],
        "intent": "view_appointments",
        "available_services": SERVICES,
    }

    with patch(
        "app.ai.agent.get_appointment_by_reference",
        return_value=APPOINTMENT,
    ) as mocked_lookup:
        result = determine_next_question(state)

    mocked_lookup.assert_called_once_with("APT-871E6728")
    response = str(result["next_question"])
    assert "Here is your appointment status:" in response
    assert "Reference: APT-871E6728" in response
    assert "Status: CONFIRMED" in response


def test_unknown_appointment_reference_returns_not_found() -> None:
    with patch(
        "app.ai.agent.get_appointment_by_reference",
        return_value=None,
    ):
        result = determine_next_question(
            {
                "messages": [
                    HumanMessage(
                        content="appointment status APT-ZZZZZZZZ",
                    ),
                ],
                "intent": "view_appointments",
                "available_services": SERVICES,
            }
        )

    assert result["next_question"] == (
        "I couldn't find an appointment with that reference. Please "
        "check the reference number or share your phone number."
    )


def test_listing_without_identity_asks_for_phone_or_reference() -> None:
    result = determine_next_question(
        {
            "messages": [
                HumanMessage(content="which appointments do I have"),
            ],
            "intent": "view_appointments",
            "available_services": SERVICES,
        }
    )

    assert result["next_question"] == (
        "Sure, I can check that. Please share your phone number or "
        "appointment reference."
    )


def test_listing_no_appointments_returns_clear_response() -> None:
    with patch(
        "app.ai.agent.get_upcoming_appointments_for_customer",
        return_value=[],
    ):
        result = determine_next_question(
            {
                "messages": [
                    HumanMessage(content="show my bookings"),
                ],
                "intent": "view_appointments",
                "customer_phone_number": "+94771234567",
                "available_services": SERVICES,
            }
        )

    assert result["next_question"] == (
        "I couldn't find any upcoming appointments for those details."
    )
