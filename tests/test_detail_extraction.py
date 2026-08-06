import pytest
from langchain_core.messages import HumanMessage
from datetime import date
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


def test_parse_requested_date_understands_ordinal_day(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.ai.agent.business_today",
        lambda: date(2026, 8, 1),
    )

    result = extract_details(
        {
            "messages": [
                HumanMessage(content="05th please"),
            ],
            "intent": "book_appointment",
        }
    )

    assert result == {
        "requested_date": "2026-08-05",
    }

def test_parse_requested_date_does_not_treat_ids_as_dates() -> None:
    customer_result = extract_details(
        {
            "messages": [
                HumanMessage(content="My customer ID is 3."),
            ],
            "intent": "book_appointment",
        }
    )

    service_result = extract_details(
        {
            "messages": [
                HumanMessage(content="I need service 2."),
            ],
            "intent": "book_appointment",
        }
    )

    appointment_result = extract_details(
        {
            "messages": [
                HumanMessage(
                    content="Cancel appointment 12 because I am travelling."
                ),
            ],
            "intent": "cancel_appointment",
        }
    )

    assert customer_result == {
        "customer_id": 3,
    }

    assert service_result == {
        "service_id": 2,
    }

    assert appointment_result == {
        "appointment_id": 12,
        "cancellation_reason": "I am travelling",
    }