from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from app.ai.agent import (
    determine_next_question,
    lookup_conversation_availability,
)


SERVICES = [
    {
        "id": 2,
        "name": "Dental care",
        "description": "Dental checkups.",
        "duration_minutes": 30,
        "price": 3500,
    }
]


def test_price_does_not_call_availability() -> None:
    with patch("app.ai.agent.check_available_slots") as availability:
        result = determine_next_question(
            {
                "messages": [HumanMessage(content="How much is dental?")],
                "intent": "general_question",
                "available_services": SERVICES,
            }
        )

    availability.assert_not_called()
    assert "LKR 3,500" in result["next_question"]


def test_duration_does_not_call_booking() -> None:
    with patch("app.ai.agent.create_appointment") as create:
        result = determine_next_question(
            {
                "messages": [HumanMessage(content="How long is dental?")],
                "intent": "general_question",
                "available_services": SERVICES,
            }
        )

    create.assert_not_called()
    assert result["next_question"] == "Dental care takes 30 minutes."


def test_availability_calls_tool_once_with_correct_arguments() -> None:
    tool = MagicMock()
    tool.run.return_value = []

    with patch("app.ai.agent.check_available_slots", tool):
        result = lookup_conversation_availability(
            {
                "messages": [HumanMessage(content="Dental tomorrow")],
                "intent": "check_availability",
                "service_id": 2,
                "staff_id": 5,
                "requested_date": "2026-08-06",
            }
        )

    tool.run.assert_called_once_with(
        {
            "service_id": 2,
            "staff_id": 5,
            "requested_date": "2026-08-06",
        }
    )
    assert result["available_slots"] == []


def test_invalid_tool_output_produces_controlled_response() -> None:
    tool = MagicMock()
    tool.run.return_value = [{"slot_id": 7, "service_id": 999}]

    with patch("app.ai.agent.check_available_slots", tool):
        updates = lookup_conversation_availability(
            {
                "messages": [HumanMessage(content="Dental tomorrow")],
                "intent": "check_availability",
                "service_id": 2,
                "requested_date": "2026-08-06",
            }
        )

    response = determine_next_question(
        {
            "messages": [HumanMessage(content="Dental tomorrow")],
            "intent": "check_availability",
            "available_services": SERVICES,
            **updates,
        }
    )
    assert "couldn't verify availability" in response["next_question"]


def test_tool_failure_does_not_invent_availability() -> None:
    tool = MagicMock()
    tool.run.side_effect = RuntimeError("database unavailable")

    with patch("app.ai.agent.check_available_slots", tool):
        result = lookup_conversation_availability(
            {
                "messages": [HumanMessage(content="Dental tomorrow")],
                "intent": "check_availability",
                "service_id": 2,
                "requested_date": "2026-08-06",
            }
        )

    assert result["available_slots"] is None
    assert "couldn't verify availability" in result["tool_error"]
