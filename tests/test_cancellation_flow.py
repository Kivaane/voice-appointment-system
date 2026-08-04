from unittest.mock import patch

from langchain_core.messages import HumanMessage

from app.ai.agent import (
    confirm_or_reject_booking,
    detect_intent,
    determine_next_question,
    extract_details,
)


APPOINTMENT = {
    "appointment_id": 12,
    "appointment_reference_number": "APT-CANCEL123",
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

SERVICES = [{"id": 2, "name": "Dental care"}]


def test_cancel_by_reference_asks_confirmation() -> None:
    initial_state = {
        "messages": [
            HumanMessage(content="cancel APT-CANCEL123"),
        ],
        "available_services": SERVICES,
    }
    intent_updates = detect_intent(initial_state)

    with patch(
        "app.ai.agent.get_appointment_by_reference",
        return_value=APPOINTMENT,
    ):
        detail_updates = extract_details(
            {
                **initial_state,
                **intent_updates,
            }
        )

    with patch(
        "app.ai.agent.get_appointment_by_id_for_conversation",
        return_value=APPOINTMENT,
    ):
        question_updates = determine_next_question(
            {
                **initial_state,
                **intent_updates,
                **detail_updates,
            }
        )

    assert intent_updates["intent"] == "cancel_appointment"
    assert "Should I cancel this appointment?" in str(
        question_updates["next_question"],
    )
    assert question_updates["confirmation_status"] == "pending"


def test_confirmed_cancellation_calls_service_helper() -> None:
    with patch(
        "app.ai.agent.cancel_confirmed_appointment_from_state",
        return_value={
            **APPOINTMENT,
            "appointment_status": "CANCELLED_BY_CUSTOMER",
        },
    ) as mocked_cancel:
        result = confirm_or_reject_booking(
            {
                "intent": "cancel_appointment",
                "appointment_id": 12,
                "appointment_reference_number": "APT-CANCEL123",
                "confirmation_status": "confirmed",
                "booking_summary": "Should I cancel this appointment?",
            }
        )

    mocked_cancel.assert_called_once()
    assert result["next_question"] == (
        "Your appointment has been cancelled. Reference: "
        "APT-CANCEL123."
    )


def test_no_keeps_appointment_without_calling_service() -> None:
    with patch(
        "app.ai.agent.cancel_confirmed_appointment_from_state",
    ) as mocked_cancel:
        result = confirm_or_reject_booking(
            {
                "intent": "cancel_appointment",
                "appointment_id": 12,
                "confirmation_status": "rejected",
                "booking_summary": "Should I cancel this appointment?",
            }
        )

    mocked_cancel.assert_not_called()
    assert result["next_question"] == (
        "No problem, I've kept your appointment as it is."
    )


def test_unknown_cancel_reference_returns_not_found() -> None:
    result = determine_next_question(
        {
            "messages": [
                HumanMessage(content="cancel APT-UNKNOWN1"),
            ],
            "intent": "cancel_appointment",
            "appointment_reference_number": "APT-UNKNOWN1",
            "available_services": SERVICES,
        }
    )
    assert result["next_question"] == (
        "I couldn't find an appointment with that reference. Please "
        "check the reference number or share your phone number."
    )


def test_cancel_without_identity_asks_for_phone_or_reference() -> None:
    result = determine_next_question(
        {
            "messages": [HumanMessage(content="cancel my appointment")],
            "intent": "cancel_appointment",
            "available_services": SERVICES,
        }
    )
    assert result["next_question"] == (
        "Sure, I can help cancel an appointment. Please share your "
        "phone number or appointment reference."
    )


def test_already_cancelled_appointment_is_reported() -> None:
    with patch(
        "app.ai.agent.get_appointment_by_id_for_conversation",
        return_value={
            **APPOINTMENT,
            "appointment_status": "CANCELLED_BY_CUSTOMER",
        },
    ):
        result = determine_next_question(
            {
                "messages": [HumanMessage(content="cancel my appointment")],
                "intent": "cancel_appointment",
                "appointment_id": 12,
                "available_services": SERVICES,
            }
        )
    assert result["next_question"] == (
        "This appointment was already cancelled."
    )


def test_stray_yes_does_not_cancel() -> None:
    with patch(
        "app.ai.agent.cancel_confirmed_appointment_from_state",
    ) as mocked_cancel:
        result = confirm_or_reject_booking(
            {
                "intent": "cancel_appointment",
                "appointment_id": 12,
                "confirmation_status": "confirmed",
                "booking_summary": None,
            }
        )

    mocked_cancel.assert_not_called()
    assert result == {}


def test_maybe_during_cancellation_asks_for_clear_confirmation() -> None:
    result = determine_next_question(
        {
            "messages": [HumanMessage(content="maybe")],
            "intent": "cancel_appointment",
            "appointment_id": 12,
            "confirmation_status": "pending",
            "booking_summary": "Should I cancel this appointment?",
            "available_services": SERVICES,
        }
    )
    assert result["next_question"] == (
        "Please reply yes to confirm, or no to cancel/change this "
        "request."
    )
