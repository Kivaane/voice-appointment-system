from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.ai.agent import (
    call_model,
    detect_intent,
    determine_next_question,
    extract_details,
    is_upcoming_availability_request,
    lookup_conversation_availability,
)


DEMO_SERVICES = [
    {
        "id": 2,
        "name": "Dental care",
        "description": "Dental checkups.",
        "duration_minutes": 30,
        "price": 3500,
    }
]

UPCOMING_SLOT = {
    "slot_id": 9,
    "service_id": 2,
    "service_name": "Dental care",
    "staff_id": 5,
    "staff_name": "Dr. Perera",
    "start_datetime": "2026-08-06T14:00:00",
    "end_datetime": "2026-08-06T14:30:00",
    "status": "AVAILABLE",
}


class FixedDate(date):
    @classmethod
    def today(cls) -> date:
        return cls(2026, 8, 4)


@pytest.mark.parametrize(
    "message",
    [
        "which date available",
        "which dates are available",
        "tell me available dates",
        "show available dates",
        "any available date",
        "when are you available",
    ],
)
def test_detects_upcoming_availability_requests(message: str) -> None:
    assert is_upcoming_availability_request(message) is True


def test_booking_flow_shows_upcoming_available_dates() -> None:
    availability_tool = MagicMock()

    def available_slots(arguments: dict[str, object]):
        if arguments["requested_date"] == "2026-08-06":
            return [UPCOMING_SLOT]

        return []

    availability_tool.run.side_effect = available_slots
    state = {
        "messages": [
            HumanMessage(content="which date available"),
        ],
        "intent": "book_appointment",
        "service_id": 2,
        "service_name": "Dental care",
        "staff_id": 5,
        "staff_name": "Dr. Perera",
        "available_services": DEMO_SERVICES,
    }

    with (
        patch("app.ai.agent.date", FixedDate),
        patch(
            "app.ai.agent.check_available_slots",
            availability_tool,
        ),
    ):
        availability_updates = lookup_conversation_availability(
            state,
        )

    assert availability_tool.run.call_count == 14
    assert availability_updates["requested_date"] is None
    assert availability_updates["available_slots"] == [UPCOMING_SLOT]

    question_updates = determine_next_question(
        {
            **state,
            **availability_updates,
        }
    )
    response = question_updates["next_question"]

    assert response is not None
    assert "I checked upcoming Dental care availability." in response
    assert "Next available slots:" in response
    assert "Thursday, 06 August 2026:" in response
    assert "1. 2:00 PM – 2:30 PM with Dr. Perera" in response

    with patch(
        "app.ai.agent.get_chat_model",
    ) as mocked_get_chat_model:
        model_result = call_model(
            {
                **state,
                **availability_updates,
                **question_updates,
            }
        )

    mocked_get_chat_model.assert_not_called()
    assert isinstance(model_result["messages"][0], AIMessage)
    assert model_result["messages"][0].content == response


def test_reschedule_flow_replaces_old_date_with_upcoming_slots() -> None:
    availability_tool = MagicMock()
    availability_tool.run.side_effect = (
        lambda arguments: (
            [UPCOMING_SLOT]
            if arguments["requested_date"] == "2026-08-06"
            else []
        )
    )
    state = {
        "messages": [
            HumanMessage(content="which date available"),
        ],
        "intent": "reschedule_appointment",
        "appointment_id": 12,
        "appointment_reference_number": "APT-871E6728",
        "service_id": 2,
        "service_name": "Dental care",
        "staff_id": 5,
        "staff_name": "Dr. Perera",
        "requested_date": "2026-08-05",
        "available_slots": [],
        "confirmation_status": "not_requested",
        "available_services": DEMO_SERVICES,
    }

    intent_updates = detect_intent(state)

    assert intent_updates["intent"] == "reschedule_appointment"

    with (
        patch("app.ai.agent.date", FixedDate),
        patch(
            "app.ai.agent.check_available_slots",
            availability_tool,
        ),
    ):
        availability_updates = lookup_conversation_availability(
            {
                **state,
                **intent_updates,
            }
        )

    assert availability_updates["requested_date"] is None
    assert availability_updates["available_slots"] == [UPCOMING_SLOT]

    question_updates = determine_next_question(
        {
            **state,
            **intent_updates,
            **availability_updates,
        }
    )

    assert "I checked upcoming Dental care availability." in str(
        question_updates["next_question"],
    )
    assert "There are no available Dental care slots" not in str(
        question_updates["next_question"],
    )


def test_no_upcoming_slots_returns_controlled_response() -> None:
    availability_tool = MagicMock()
    availability_tool.run.return_value = []
    state = {
        "messages": [
            HumanMessage(content="show available dates"),
        ],
        "intent": "book_appointment",
        "service_id": 2,
        "service_name": "Dental care",
        "available_services": DEMO_SERVICES,
    }

    with (
        patch("app.ai.agent.date", FixedDate),
        patch(
            "app.ai.agent.check_available_slots",
            availability_tool,
        ),
    ):
        availability_updates = lookup_conversation_availability(
            state,
        )

    question_updates = determine_next_question(
        {
            **state,
            **availability_updates,
        }
    )

    assert question_updates["next_question"] == (
        "I checked upcoming Dental care availability, but there are "
        "no available slots in the current demo data. You can try "
        "another service or ask staff to add more availability."
    )

    with patch(
        "app.ai.agent.get_chat_model",
    ) as mocked_get_chat_model:
        result = call_model(
            {
                **state,
                **availability_updates,
                **question_updates,
            }
        )

    mocked_get_chat_model.assert_not_called()
    assert result["messages"][0].content == (
        question_updates["next_question"]
    )


def test_selecting_upcoming_slot_updates_requested_date() -> None:
    updates = extract_details(
        {
            "messages": [
                HumanMessage(content="first one"),
            ],
            "intent": "reschedule_appointment",
            "appointment_id": 12,
            "available_slots": [UPCOMING_SLOT],
            "confirmation_status": "not_requested",
        }
    )

    assert updates["slot_id"] == 9
    assert updates["requested_date"] == "2026-08-06"
