import pytest
from langchain_core.messages import HumanMessage

from app.ai.agent import extract_details


@pytest.mark.parametrize(
    ("message", "intent", "expected"),
    [
        (
            "My customer ID is 3.",
            "book_appointment",
            {
                "customer_id": 3,
            },
        ),
        (
            "I need service 2.",
            "book_appointment",
            {
                "service_id": 2,
            },
        ),
        (
            "Please use staff ID 5 and slot 7.",
            "book_appointment",
            {
                "staff_id": 5,
                "slot_id": 7,
            },
        ),
        (
            "I need service 2 on 2026-08-05.",
            "check_availability",
            {
                "service_id": 2,
                "requested_date": "2026-08-05",
            },
        ),
        (
            "Cancel appointment 12 because I am travelling.",
            "cancel_appointment",
            {
                "appointment_id": 12,
                "cancellation_reason": "I am travelling",
            },
        ),
        (
            "Move appointment ID 9 to slot 20.",
            "reschedule_appointment",
            {
                "appointment_id": 9,
                "slot_id": 20,
            },
        ),
    ],
)
def test_extracts_appointment_details(
    message: str,
    intent: str,
    expected: dict[str, object],
) -> None:
    result = extract_details(
        {
            "messages": [
                HumanMessage(content=message),
            ],
            "intent": intent,
        }
    )

    assert result == expected


def test_returns_empty_update_when_no_details_exist() -> None:
    result = extract_details(
        {
            "messages": [
                HumanMessage(
                    content="I would like some help."
                ),
            ],
            "intent": "general_question",
        }
    )

    assert result == {}