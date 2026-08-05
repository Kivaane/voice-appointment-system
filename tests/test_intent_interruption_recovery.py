from langchain_core.messages import HumanMessage

from app.ai.agent import detect_intent, determine_next_question


SERVICES = [
    {
        "id": 2,
        "name": "Dental care",
        "description": "Dental checkups.",
        "duration_minutes": 30,
        "price": 3500,
    }
]


def answer_interruption(message: str, **state_values):
    state = {
        "messages": [HumanMessage(content=message)],
        "available_services": SERVICES,
        **state_values,
    }
    intent_updates = detect_intent(state)
    updated_state = {**state, **intent_updates}
    question_updates = determine_next_question(updated_state)
    return {**updated_state, **question_updates}


def test_price_question_during_booking_preserves_booking_state() -> None:
    result = answer_interruption(
        "How much is dental?",
        intent="book_appointment",
        service_id=2,
        service_name="Dental care",
        requested_date="2026-08-05",
    )

    assert result["intent"] == "book_appointment"
    assert result["service_id"] == 2
    assert result["requested_date"] == "2026-08-05"
    assert result["next_question"] == (
        "Dental care costs LKR 3,500. You were selecting a slot for "
        "Wednesday, 05 August 2026. Would you like me to continue?"
    )


def test_duration_question_during_booking_preserves_booking_state() -> None:
    result = answer_interruption(
        "How long is dental?",
        intent="book_appointment",
        service_id=2,
        service_name="Dental care",
        requested_date="2026-08-05",
    )

    assert result["intent"] == "book_appointment"
    assert result["service_id"] == 2
    assert result["next_question"] == (
        "Dental care takes 30 minutes. You were selecting a slot for "
        "Wednesday, 05 August 2026. Would you like me to continue?"
    )


def test_opening_hours_during_reschedule_preserves_state() -> None:
    result = answer_interruption(
        "What time are you open?",
        intent="reschedule_appointment",
        appointment_id=17,
        appointment_reference_number="APT-TEST",
        requested_date="2026-08-06",
    )

    assert result["intent"] == "reschedule_appointment"
    assert result["appointment_id"] == 17
    assert result["requested_date"] == "2026-08-06"
    assert "front desk" in result["next_question"]


def test_policy_question_during_cancellation_preserves_confirmation() -> None:
    result = answer_interruption(
        "What is your cancellation policy?",
        intent="cancel_appointment",
        appointment_id=21,
        booking_summary="Should I cancel this appointment?",
        confirmation_status="pending",
    )

    assert result["intent"] == "cancel_appointment"
    assert result["appointment_id"] == 21
    assert result["confirmation_status"] == "pending"
    assert result["booking_summary"] == "Should I cancel this appointment?"


def test_next_transactional_message_continues_after_interruption() -> None:
    interrupted = answer_interruption(
        "How much is dental?",
        intent="book_appointment",
        service_id=2,
        service_name="Dental care",
        requested_date="2026-08-05",
    )
    continuation_state = {
        **interrupted,
        "messages": [HumanMessage(content="Show me the slots")],
    }

    intent_updates = detect_intent(continuation_state)

    assert intent_updates["intent"] == "book_appointment"
    assert continuation_state["service_id"] == 2
    assert continuation_state["requested_date"] == "2026-08-05"


def test_information_question_preserves_availability_flow() -> None:
    result = answer_interruption(
        "What payment methods do you accept?",
        intent="check_availability",
        service_id=2,
        requested_date="2026-08-05",
    )

    assert result["intent"] == "check_availability"
    assert result["service_id"] == 2
    assert result["requested_date"] == "2026-08-05"
