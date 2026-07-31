from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.ai.agent import appointment_agent


def test_booking_conversation_preserves_state() -> None:
    thread_id = "booking-conversation-test-001"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    mocked_tool = MagicMock()
    mocked_tool.run.return_value = [
        {
            "slot_id": 7,
            "service_id": 2,
            "staff_id": 5,
            "start_datetime": "2026-08-05T10:00:00",
            "end_datetime": "2026-08-05T10:30:00",
            "status": "available",
        }
    ]

    with patch(
        "app.ai.agent.check_available_slots",
        mocked_tool,
    ):
        first_result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="I need an appointment."
                    )
                ]
            },
            config=config,
        )

        first_response = first_result["messages"][-1]

        assert isinstance(first_response, AIMessage)
        assert first_response.content == (
            "Which service would you like to book?"
        )

        assert first_result["intent"] == "book_appointment"
        assert first_result.get("service_id") is None

        second_result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Service 2.")
                ]
            },
            config=config,
        )

        second_response = second_result["messages"][-1]

        assert isinstance(second_response, AIMessage)
        assert second_response.content == (
            "Which date would you prefer?"
        )

        assert second_result["intent"] == "book_appointment"
        assert second_result.get("service_id") == 2

        third_result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="2026-08-05")
                ]
            },
            config=config,
        )

    third_response = third_result["messages"][-1]

    assert isinstance(third_response, AIMessage)

    assert third_response.content == (
        "These appointment slots are available:\n"
        "7: 2026-08-05T10:00:00 "
        "to 2026-08-05T10:30:00 "
        "with staff 5\n"
        "Which slot would you prefer?"
    )

    assert third_result["intent"] == "book_appointment"
    assert third_result.get("service_id") == 2
    assert (
        third_result.get("requested_date")
        == "2026-08-05"
    )
    assert third_result.get("available_slots") == [
        {
            "slot_id": 7,
            "service_id": 2,
            "staff_id": 5,
            "start_datetime": "2026-08-05T10:00:00",
            "end_datetime": "2026-08-05T10:30:00",
            "status": "available",
        }
    ]

    mocked_tool.run.assert_called_once_with(
        {
            "service_id": 2,
            "requested_date": "2026-08-05",
        }
    )