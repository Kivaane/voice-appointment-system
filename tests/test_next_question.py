import pytest

from app.ai.agent import determine_next_question


@pytest.mark.parametrize(
    ("state", "expected_question"),
    [
        (
            {
                "intent": "book_appointment",
            },
            "Which service would you like to book?",
        ),
        (
            {
                "intent": "book_appointment",
                "service_id": 2,
            },
            "Which date would you prefer?",
        ),
        (
            {
                "intent": "book_appointment",
                "service_id": 2,
                "requested_date": "2026-08-05",
            },
            "Which available appointment slot would you prefer?",
        ),
        (
            {
                "intent": "book_appointment",
                "service_id": 2,
                "requested_date": "2026-08-05",
                "slot_id": 7,
            },
            "What is your customer ID?",
        ),
        (
            {
                "intent": "check_availability",
            },
            "Which service would you like to check?",
        ),
        (
            {
                "intent": "check_availability",
                "service_id": 2,
            },
            "Which date would you like to check?",
        ),
        (
            {
                "intent": "cancel_appointment",
            },
            "What is your appointment ID?",
        ),
        (
            {
                "intent": "cancel_appointment",
                "appointment_id": 12,
            },
            "What is the reason for the cancellation?",
        ),
        (
            {
                "intent": "reschedule_appointment",
            },
            "What is your appointment ID?",
        ),
        (
            {
                "intent": "reschedule_appointment",
                "appointment_id": 12,
            },
            "Which new appointment slot would you prefer?",
        ),
        (
            {
                "intent": "general_question",
            },
            None,
        ),
    ],
)
def test_determines_next_question(
    state,
    expected_question,
) -> None:
    result = determine_next_question(state)

    assert result["next_question"] == expected_question