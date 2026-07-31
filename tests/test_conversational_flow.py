from langchain_core.messages import AIMessage, HumanMessage

from app.ai.agent import appointment_agent


def test_booking_conversation_preserves_state() -> None:
    thread_id = "booking-conversation-test-001"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

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
        "Which available appointment slot would you prefer?"
    )

    assert third_result["intent"] == "book_appointment"
    assert third_result.get("service_id") == 2
    assert (
        third_result.get("requested_date")
        == "2026-08-05"
    )