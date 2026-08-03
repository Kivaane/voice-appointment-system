from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.ai.agent import appointment_agent, extract_text_content


DEMO_SERVICES = [
    {
        "id": 2,
        "name": "Dental care",
        "description": "Dental checkups.",
        "duration_minutes": 30,
        "price": 3500,
    }
]


DEMO_STAFF = [
    {
        "id": 5,
        "full_name": "Dr. Perera",
        "speciality": "Dental care",
    }
]


DEMO_SLOTS = [
    {
        "slot_id": 7,
        "service_id": 2,
        "service_name": "Dental care",
        "staff_id": 5,
        "staff_name": "Dr. Perera",
        "start_datetime": "2026-08-05T10:00:00",
        "end_datetime": "2026-08-05T10:30:00",
        "status": "AVAILABLE",
    }
]


def test_booking_conversation_preserves_state() -> None:
    thread_id = "booking-conversation-natural-flow-test-001"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    mocked_tool = MagicMock()
    mocked_tool.run.return_value = DEMO_SLOTS

    with (
        patch(
            "app.ai.agent.get_active_services",
            return_value=DEMO_SERVICES,
        ),
        patch(
            "app.ai.agent.get_active_staff_for_service",
            return_value=DEMO_STAFF,
        ),
        patch(
            "app.ai.agent.check_available_slots",
            mocked_tool,
        ),
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
        assert "Sure, I can help you book an appointment." in (
            first_response.content
        )
        assert "Which service would you like?" in (
            first_response.content
        )
        assert "Dental care" in first_response.content

        assert first_result["intent"] == "book_appointment"
        assert first_result.get("service_id") is None

        second_result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Dental care.")
                ]
            },
            config=config,
        )

        second_response = second_result["messages"][-1]

        assert isinstance(second_response, AIMessage)
        assert "Sure, I can help you book Dental care." in (
            second_response.content
        )
        assert "Dr. Perera" in second_response.content
        assert "Which date would you prefer?" in (
            second_response.content
        )

        assert second_result["intent"] == "book_appointment"
        assert second_result.get("service_id") == 2
        assert second_result.get("service_name") == "Dental care"
        assert second_result.get("staff_id") == 5
        assert second_result.get("staff_name") == "Dr. Perera"

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
        assert "I found these Dental care slots" in (
            third_response.content
        )
        assert "Wednesday, 05 August 2026" in (
            third_response.content
        )
        assert "1. 10:00 AM – 10:30 AM with Dr. Perera" in (
            third_response.content
        )
        assert "Which one would you prefer?" in (
            third_response.content
        )

        assert third_result["intent"] == "book_appointment"
        assert third_result.get("service_id") == 2
        assert third_result.get("requested_date") == "2026-08-05"
        assert third_result.get("available_slots") == DEMO_SLOTS

        fourth_result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="first one")
                ]
            },
            config=config,
        )

        fourth_response = fourth_result["messages"][-1]

        assert isinstance(fourth_response, AIMessage)
        assert "Great. You selected Dental care at 10:00 AM" in (
            fourth_response.content
        )
        assert "with Dr. Perera" in fourth_response.content
        assert "full name and phone number" in fourth_response.content

        assert fourth_result.get("slot_id") == 7
        assert fourth_result.get("staff_id") == 5
        assert fourth_result.get("staff_name") == "Dr. Perera"

    mocked_tool.run.assert_called_once_with(
        {
            "service_id": 2,
            "requested_date": "2026-08-05",
            "staff_id": 5,
        }
    )


def test_booking_conversation_collects_customer_and_shows_summary() -> None:
    thread_id = "booking-conversation-customer-summary-test-001"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    mocked_tool = MagicMock()
    mocked_tool.run.return_value = DEMO_SLOTS

    with (
        patch(
            "app.ai.agent.get_active_services",
            return_value=DEMO_SERVICES,
        ),
        patch(
            "app.ai.agent.get_active_staff_for_service",
            return_value=DEMO_STAFF,
        ),
        patch(
            "app.ai.agent.check_available_slots",
            mocked_tool,
        ),
        patch(
            "app.ai.agent.get_or_create_customer_from_details",
            return_value={
                "id": 11,
                "full_name": "Kivaane Anton",
                "phone_number": "0774588691",
            },
        ) as mocked_customer_lookup,
    ):
        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="I need an appointment."
                    )
                ]
            },
            config=config,
        )

        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Dental care.")
                ]
            },
            config=config,
        )

        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="2026-08-05")
                ]
            },
            config=config,
        )

        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="first one")
                ]
            },
            config=config,
        )

        result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Kivaane Anton and contact number "
                            "0774588691"
                        )
                    )
                ]
            },
            config=config,
        )

        final_response = result["messages"][-1]
        final_text = extract_text_content(final_response.content)

        assert isinstance(final_response, AIMessage)

        assert "Please confirm your appointment:" in final_text
        assert "Service: Dental care" in final_text
        assert "Doctor/Staff: Dr. Perera" in final_text
        assert "Date: Wednesday, 05 August 2026" in final_text
        assert "Time: 10:00 AM – 10:30 AM" in final_text
        assert "Name: Kivaane Anton" in final_text
        assert "Phone: 0774588691" in final_text
        assert "Should I confirm this booking?" in final_text

        assert result.get("customer_id") == 11
        assert result.get("customer_name") == "Kivaane Anton"
        assert result.get("customer_phone_number") == "0774588691"
        assert result.get("confirmation_status") == "pending"

    mocked_customer_lookup.assert_called_once_with(
        full_name="Kivaane Anton",
        phone_number="0774588691",
    )


def test_booking_conversation_confirms_real_appointment() -> None:
    thread_id = "booking-conversation-confirm-appointment-test-001"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    mocked_tool = MagicMock()
    mocked_tool.run.return_value = DEMO_SLOTS

    with (
        patch(
            "app.ai.agent.get_active_services",
            return_value=DEMO_SERVICES,
        ),
        patch(
            "app.ai.agent.get_active_staff_for_service",
            return_value=DEMO_STAFF,
        ),
        patch(
            "app.ai.agent.check_available_slots",
            mocked_tool,
        ),
        patch(
            "app.ai.agent.get_or_create_customer_from_details",
            return_value={
                "id": 11,
                "full_name": "Kivaane Anton",
                "phone_number": "0774588691",
            },
        ),
        patch(
            "app.ai.agent.create_confirmed_appointment_from_state",
            return_value={
                "id": 21,
                "reference_number": "APT-TEST123",
                "start_datetime": "2026-08-05T10:00:00",
                "end_datetime": "2026-08-05T10:30:00",
            },
        ) as mocked_create_appointment,
    ):
        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="I need an appointment."
                    )
                ]
            },
            config=config,
        )

        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Dental care.")
                ]
            },
            config=config,
        )

        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="2026-08-05")
                ]
            },
            config=config,
        )

        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="first one")
                ]
            },
            config=config,
        )

        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Kivaane Anton and contact number "
                            "0774588691"
                        )
                    )
                ]
            },
            config=config,
        )

        result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="yes confirm")
                ]
            },
            config=config,
        )

        final_response = result["messages"][-1]
        final_text = extract_text_content(final_response.content)

        assert isinstance(final_response, AIMessage)

        assert "Your appointment is confirmed." in final_text
        assert "Reference: APT-TEST123" in final_text
        assert "Service: Dental care" in final_text
        assert "Doctor/Staff: Dr. Perera" in final_text
        assert "Date: Wednesday, 05 August 2026" in final_text
        assert "Time: 10:00 AM – 10:30 AM" in final_text

        assert result.get("appointment_id") == 21
        assert result.get("appointment_reference_number") == "APT-TEST123"
        assert result.get("confirmation_status") == "confirmed"

    mocked_create_appointment.assert_called_once()