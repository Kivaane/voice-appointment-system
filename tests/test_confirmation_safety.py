from unittest.mock import patch

from langchain_core.messages import HumanMessage

from app.ai.agent import (
    confirm_or_reject_booking,
    detect_intent,
    determine_next_question,
)


def test_stray_yes_does_not_create_appointment() -> None:
    with patch(
        "app.ai.agent.create_confirmed_appointment_from_state",
    ) as mocked_create:
        result = confirm_or_reject_booking(
            {
                "intent": "book_appointment",
                "customer_id": 11,
                "service_id": 2,
                "staff_id": 5,
                "slot_id": 7,
                "confirmation_status": "confirmed",
                "booking_summary": None,
            }
        )

    mocked_create.assert_not_called()
    assert result == {}


def test_maybe_asks_for_clear_confirmation() -> None:
    result = determine_next_question(
        {
            "messages": [HumanMessage(content="maybe")],
            "intent": "book_appointment",
            "confirmation_status": "pending",
            "booking_summary": "Please confirm your appointment",
            "available_services": [{"id": 2, "name": "Dental care"}],
        }
    )
    assert result["next_question"] == (
        "Please reply yes to confirm, or no to cancel/change this "
        "request."
    )


def test_no_at_booking_confirmation_does_not_create() -> None:
    state = {
        "messages": [HumanMessage(content="no")],
        "intent": "book_appointment",
        "confirmation_status": "pending",
        "booking_summary": "Please confirm your appointment",
    }
    intent_updates = detect_intent(state)

    with patch(
        "app.ai.agent.create_confirmed_appointment_from_state",
    ) as mocked_create:
        result = confirm_or_reject_booking(
            {
                **state,
                **intent_updates,
            }
        )

    mocked_create.assert_not_called()
    assert intent_updates["confirmation_status"] == "rejected"
    assert "not booked" in str(result["next_question"])


def test_no_at_reschedule_keeps_original_appointment() -> None:
    state = {
        "messages": [HumanMessage(content="no")],
        "intent": "reschedule_appointment",
        "appointment_id": 12,
        "slot_id": 9,
        "confirmation_status": "pending",
        "booking_summary": "Please confirm your rescheduled appointment",
    }
    intent_updates = detect_intent(state)

    with patch(
        "app.ai.agent.reschedule_confirmed_appointment_from_state",
    ) as mocked_reschedule:
        result = confirm_or_reject_booking(
            {
                **state,
                **intent_updates,
            }
        )

    mocked_reschedule.assert_not_called()
    assert result["next_question"] == (
        "No problem. Your original appointment stays as is."
    )


def test_reschedule_same_current_slot_does_not_call_service() -> None:
    with patch(
        "app.ai.agent.reschedule_confirmed_appointment_from_state",
    ) as mocked_reschedule:
        result = confirm_or_reject_booking(
            {
                "intent": "reschedule_appointment",
                "appointment_id": 12,
                "current_slot_id": 7,
                "slot_id": 7,
                "confirmation_status": "confirmed",
                "booking_summary": (
                    "Please confirm your rescheduled appointment"
                ),
            }
        )

    mocked_reschedule.assert_not_called()
    assert result["next_question"] == (
        "That's already your current appointment time, so no changes "
        "are needed."
    )
