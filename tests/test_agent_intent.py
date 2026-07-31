import pytest
from langchain_core.messages import HumanMessage

from app.ai.agent import detect_intent


@pytest.mark.parametrize(
    ("message", "expected_intent"),
    [
        (
            "I need an appointment.",
            "book_appointment",
        ),
        (
            "Please book a dental appointment.",
            "book_appointment",
        ),
        (
            "Cancel my appointment.",
            "cancel_appointment",
        ),
        (
            "I need to reschedule my appointment.",
            "reschedule_appointment",
        ),
        (
            "What available slots do you have?",
            "check_availability",
        ),
        (
            "What services do you offer?",
            "list_services",
        ),
        (
            "Hello, how are you?",
            "general_question",
        ),
    ],
)
def test_detects_appointment_intent(
    message: str,
    expected_intent: str,
) -> None:
    result = detect_intent(
        {
            "messages": [
                HumanMessage(content=message),
            ]
        }
    )

    assert result["intent"] == expected_intent


def test_preserves_booking_intent_for_follow_up_details() -> None:
    result = detect_intent(
        {
            "messages": [
                HumanMessage(content="Tomorrow morning."),
            ],
            "intent": "book_appointment",
        }
    )

    assert result["intent"] == "book_appointment"


def test_preserves_rescheduling_intent_for_follow_up_details() -> None:
    result = detect_intent(
        {
            "messages": [
                HumanMessage(content="Friday at 2 PM."),
            ],
            "intent": "reschedule_appointment",
        }
    )

    assert result["intent"] == "reschedule_appointment"