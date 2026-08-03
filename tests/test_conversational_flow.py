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
    },
    {
        "slot_id": 8,
        "service_id": 2,
        "service_name": "Dental care",
        "staff_id": 5,
        "staff_name": "Dr. Perera",
        "start_datetime": "2026-08-05T14:30:00",
        "end_datetime": "2026-08-05T15:00:00",
        "status": "AVAILABLE",
    },
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


def test_booking_conversation_allows_time_change_before_confirmation() -> None:
    thread_id = "booking-conversation-change-time-test-001"

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
                    HumanMessage(content="I need an appointment.")
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
                    HumanMessage(content="no, change the time")
                ]
            },
            config=config,
        )

        final_response = result["messages"][-1]
        final_text = extract_text_content(final_response.content)

        assert isinstance(final_response, AIMessage)

        assert "I found these Dental care slots" in final_text
        assert "Wednesday, 05 August 2026" in final_text
        assert "1. 10:00 AM – 10:30 AM with Dr. Perera" in final_text
        assert "Which one would you prefer?" in final_text

        assert result.get("slot_id") is None
        assert result.get("selected_slot_summary") is None
        assert result.get("confirmation_status") == "not_requested"
        assert result.get("customer_id") == 11

    mocked_create_appointment.assert_not_called()

def test_booking_conversation_allows_time_change_after_rejection() -> None:
    thread_id = "booking-conversation-change-time-after-reject-test-001"

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
                    HumanMessage(content="I need an appointment.")
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

        rejected_result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="no")
                ]
            },
            config=config,
        )

        assert rejected_result.get("confirmation_status") == "rejected"

        result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="change the time")
                ]
            },
            config=config,
        )

        final_response = result["messages"][-1]
        final_text = extract_text_content(final_response.content)

        assert isinstance(final_response, AIMessage)

        assert "I found these Dental care slots" in final_text
        assert "Wednesday, 05 August 2026" in final_text
        assert "Which one would you prefer?" in final_text

        assert result.get("slot_id") is None
        assert result.get("selected_slot_summary") is None
        assert result.get("confirmation_status") == "not_requested"
        assert result.get("customer_id") == 11

    mocked_create_appointment.assert_not_called()

def test_booking_conversation_allows_date_change_after_rejection() -> None:
    thread_id = "booking-conversation-change-date-after-reject-test-001"

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
    ):
        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="I need an appointment.")
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

        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="no")
                ]
            },
            config=config,
        )

        date_change_result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="different date")
                ]
            },
            config=config,
        )

        date_change_text = extract_text_content(
            date_change_result["messages"][-1].content
        )

        assert "Which date would you prefer?" in date_change_text
        assert date_change_result.get("requested_date") is None
        assert date_change_result.get("slot_id") is None
        assert date_change_result.get("confirmation_status") == (
            "not_requested"
        )

        new_date_result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="05th please")
                ]
            },
            config=config,
        )

        new_date_text = extract_text_content(
            new_date_result["messages"][-1].content
        )

        assert "I found these Dental care slots" in new_date_text
        assert "Wednesday, 05 August 2026" in new_date_text
        assert "Which one would you prefer?" in new_date_text
        assert new_date_result.get("requested_date") == "2026-08-05"


def test_booking_conversation_allows_service_change_after_rejection() -> None:
    thread_id = "booking-conversation-change-service-after-reject-test-001"

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
    ):
        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="I need an appointment.")
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

        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="no")
                ]
            },
            config=config,
        )

        result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="different service")
                ]
            },
            config=config,
        )

        final_text = extract_text_content(
            result["messages"][-1].content
        )

        assert "Which service would you like?" in final_text
        assert "Available services:" in final_text
        assert "Dental care" in final_text

        assert result.get("service_id") is None
        assert result.get("service_name") is None
        assert result.get("requested_date") is None
        assert result.get("slot_id") is None
        assert result.get("confirmation_status") == "not_requested"


def test_booking_conversation_allows_second_slot_before_confirmation() -> None:
    thread_id = "booking-conversation-second-slot-test-001"

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
    ):
        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="I need an appointment.")
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
                    HumanMessage(content="use the second slot instead")
                ]
            },
            config=config,
        )

        final_text = extract_text_content(
            result["messages"][-1].content
        )

        assert "Please confirm your appointment:" in final_text
        assert "Service: Dental care" in final_text
        assert "Doctor/Staff: Dr. Perera" in final_text
        assert "Date: Wednesday, 05 August 2026" in final_text
        assert "Time: 2:30 PM – 3:00 PM" in final_text
        assert "Name: Kivaane Anton" in final_text
        assert "Phone: 0774588691" in final_text

        assert result.get("slot_id") == 8
        assert result.get("selected_slot_summary") == (
            "Dental care at 2:30 PM with Dr. Perera"
        )
        assert result.get("confirmation_status") == "pending"

def test_booking_conversation_treats_1st_as_slot_choice_not_date() -> None:
    thread_id = "booking-conversation-1st-slot-not-date-test-001"

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
        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="I need a dental appointment.")
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

        result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="1st")
                ]
            },
            config=config,
        )

        final_text = extract_text_content(
            result["messages"][-1].content
        )

        assert "Great. You selected Dental care at 10:00 AM" in final_text
        assert "May I have your full name and phone number" in final_text
        assert "September" not in final_text

        assert result.get("requested_date") == "2026-08-05"
        assert result.get("slot_id") == 7
        assert result.get("selected_slot_summary") == (
            "Dental care at 10:00 AM with Dr. Perera"
        )


def test_booking_conversation_starts_new_booking_after_confirmed_appointment() -> None:
    thread_id = "booking-conversation-new-booking-after-confirmed-test-001"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    services = [
        {
            "id": 2,
            "name": "Dental care",
            "description": "Dental checkups and treatment appointments.",
            "duration_minutes": 30,
            "price": 3500.00,
        },
        {
            "id": 7,
            "name": "Physiotherapy",
            "description": "Physiotherapy treatment appointment.",
            "duration_minutes": 45,
            "price": 4500.00,
        },
    ]

    dental_staff = [
        {
            "id": 5,
            "full_name": "Dr. Perera",
            "speciality": "Dental care",
        }
    ]

    physiotherapy_staff = [
        {
            "id": 6,
            "full_name": "Therapist Nimal",
            "speciality": "Physiotherapy",
        }
    ]

    physiotherapy_slots = [
        {
            "slot_id": 20,
            "service_id": 7,
            "service_name": "Physiotherapy",
            "staff_id": 6,
            "staff_name": "Therapist Nimal",
            "start_datetime": "2026-08-06T10:30:00",
            "end_datetime": "2026-08-06T11:15:00",
            "status": "AVAILABLE",
        }
    ]

    mocked_tool = MagicMock()

    def check_slots(arguments: dict[str, object]) -> list[dict[str, object]]:
        if arguments["service_id"] == 7:
            return physiotherapy_slots

        return DEMO_SLOTS

    def get_staff(service_id: int) -> list[dict[str, object]]:
        if service_id == 7:
            return physiotherapy_staff

        return dental_staff

    mocked_tool.run.side_effect = check_slots

    with (
        patch(
            "app.ai.agent.get_active_services",
            return_value=services,
        ),
        patch(
            "app.ai.agent.get_active_staff_for_service",
            side_effect=get_staff,
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
            side_effect=[
                {
                    "id": 101,
                    "reference_number": "APT-DENTAL-001",
                    "start_datetime": "2026-08-05T10:00:00",
                    "end_datetime": "2026-08-05T10:30:00",
                },
                {
                    "id": 102,
                    "reference_number": "APT-PHYSIO-001",
                    "start_datetime": "2026-08-06T10:30:00",
                    "end_datetime": "2026-08-06T11:15:00",
                },
            ],
        ),
    ):
        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="I need a dental appointment.")
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

        first_booking_result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="yes confirm")
                ]
            },
            config=config,
        )

        assert first_booking_result.get("appointment_id") == 101
        assert first_booking_result.get("appointment_reference_number") == (
            "APT-DENTAL-001"
        )

        new_booking_result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="I want to book Physiotherapy")
                ]
            },
            config=config,
        )

        new_booking_text = extract_text_content(
            new_booking_result["messages"][-1].content
        )

        assert "Which date would you prefer?" in new_booking_text
        assert new_booking_result.get("appointment_id") is None
        assert new_booking_result.get("appointment_reference_number") is None
        assert new_booking_result.get("service_id") == 7
        assert new_booking_result.get("service_name") == "Physiotherapy"
        assert new_booking_result.get("slot_id") is None
        assert new_booking_result.get("confirmation_status") == (
            "not_requested"
        )

        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="2026-08-06")
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

        second_booking_result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="yes confirm")
                ]
            },
            config=config,
        )

        second_booking_text = extract_text_content(
            second_booking_result["messages"][-1].content
        )

        assert "Your appointment is confirmed." in second_booking_text
        assert "APT-PHYSIO-001" in second_booking_text
        assert "Physiotherapy" in second_booking_text
        assert "Therapist Nimal" in second_booking_text

        assert second_booking_result.get("appointment_id") == 102
        assert second_booking_result.get("appointment_reference_number") == (
            "APT-PHYSIO-001"
        )
        assert second_booking_result.get("service_id") == 7
        assert second_booking_result.get("slot_id") == 20

def test_service_numeric_choice_uses_display_option_before_database_id() -> None:
    thread_id = "booking-conversation-service-option-number-test-001"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    services = [
        {
            "id": 4,
            "name": "Dental care",
            "description": "Dental checkups and treatment appointments.",
            "duration_minutes": 30,
            "price": 3500.00,
        },
        {
            "id": 5,
            "name": "General consultation",
            "description": "General medical consultation appointment.",
            "duration_minutes": 20,
            "price": 2500.00,
        },
        {
            "id": 6,
            "name": "Dermatology",
            "description": "Skin care consultation appointment.",
            "duration_minutes": 30,
            "price": 4000.00,
        },
        {
            "id": 7,
            "name": "Physiotherapy",
            "description": "Physiotherapy treatment appointment.",
            "duration_minutes": 45,
            "price": 4500.00,
        },
    ]

    def get_staff(service_id: int) -> list[dict[str, object]]:
        if service_id == 7:
            return [
                {
                    "id": 6,
                    "full_name": "Therapist Nimal",
                    "speciality": "Physiotherapy",
                }
            ]

        return [
            {
                "id": 1,
                "full_name": "Dr. Perera",
                "speciality": "Dental care",
            }
        ]

    with (
        patch(
            "app.ai.agent.get_active_services",
            return_value=services,
        ),
        patch(
            "app.ai.agent.get_active_staff_for_service",
            side_effect=get_staff,
        ),
    ):
        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="I want to book another appointment")
                ]
            },
            config=config,
        )

        result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="4")
                ]
            },
            config=config,
        )

        final_text = extract_text_content(
            result["messages"][-1].content
        )

        assert "Physiotherapy" in final_text
        assert "Therapist Nimal" in final_text
        assert "Dental care" not in final_text

        assert result.get("service_id") == 7
        assert result.get("service_name") == "Physiotherapy"

def test_booking_conversation_does_not_repeat_confirmation_after_thank_you() -> None:
    thread_id = "booking-conversation-thank-you-after-confirmed-test-001"

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
                "id": 101,
                "reference_number": "APT-DENTAL-001",
                "start_datetime": "2026-08-05T10:00:00",
                "end_datetime": "2026-08-05T10:30:00",
            },
        ),
    ):
        appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="I need a dental appointment.")
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

        confirmed_result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="yes confirm")
                ]
            },
            config=config,
        )

        confirmed_text = extract_text_content(
            confirmed_result["messages"][-1].content
        )

        assert "Your appointment is confirmed." in confirmed_text
        assert "APT-DENTAL-001" in confirmed_text

        thank_you_result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="oki thank you")
                ]
            },
            config=config,
        )

        thank_you_text = extract_text_content(
            thank_you_result["messages"][-1].content
        )

        assert "You're welcome" in thank_you_text
        assert "Your appointment is confirmed." not in thank_you_text
        assert "APT-DENTAL-001" not in thank_you_text