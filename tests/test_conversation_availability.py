from unittest.mock import MagicMock, patch

from app.ai.agent import lookup_conversation_availability


def test_checks_availability_when_service_and_date_exist() -> None:
    slots = [
        {
            "slot_id": 7,
            "service_id": 2,
            "staff_id": 5,
            "start_datetime": "2026-08-05T10:00:00",
            "end_datetime": "2026-08-05T10:30:00",
            "status": "available",
        }
    ]

    mocked_tool = MagicMock()
    mocked_tool.run.return_value = slots

    with patch(
        "app.ai.agent.check_available_slots",
        mocked_tool,
    ):
        result = lookup_conversation_availability(
            {
                "intent": "book_appointment",
                "service_id": 2,
                "requested_date": "2026-08-05",
            }
        )

    mocked_tool.run.assert_called_once_with(
        {
            "service_id": 2,
            "requested_date": "2026-08-05",
        }
    )

    assert result["available_slots"] == slots


def test_includes_staff_filter_when_staff_exists() -> None:
    mocked_tool = MagicMock()
    mocked_tool.run.return_value = []

    with patch(
        "app.ai.agent.check_available_slots",
        mocked_tool,
    ):
        result = lookup_conversation_availability(
            {
                "intent": "check_availability",
                "service_id": 2,
                "staff_id": 5,
                "requested_date": "2026-08-05",
            }
        )

    mocked_tool.run.assert_called_once_with(
        {
            "service_id": 2,
            "requested_date": "2026-08-05",
            "staff_id": 5,
        }
    )

    assert result["available_slots"] == []


def test_does_not_check_without_required_details() -> None:
    mocked_tool = MagicMock()

    with patch(
        "app.ai.agent.check_available_slots",
        mocked_tool,
    ):
        result = lookup_conversation_availability(
            {
                "intent": "book_appointment",
                "service_id": 2,
            }
        )

    mocked_tool.run.assert_not_called()
    assert result == {}


def test_does_not_check_for_unrelated_intent() -> None:
    mocked_tool = MagicMock()

    with patch(
        "app.ai.agent.check_available_slots",
        mocked_tool,
    ):
        result = lookup_conversation_availability(
            {
                "intent": "cancel_appointment",
                "service_id": 2,
                "requested_date": "2026-08-05",
            }
        )

    mocked_tool.run.assert_not_called()
    assert result == {}