from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.ai.agent import (
    appointment_agent,
    detect_intent,
    determine_next_question,
    extract_details,
    extract_text_content,
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

DEMO_STAFF = [
    {
        "id": 5,
        "full_name": "Dr. Perera",
        "speciality": "Dental care",
    }
]

OLD_SLOTS = [
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

NEW_SLOTS = [
    {
        "slot_id": 9,
        "service_id": 2,
        "service_name": "Dental care",
        "staff_id": 5,
        "staff_name": "Dr. Perera",
        "start_datetime": "2026-08-06T14:00:00",
        "end_datetime": "2026-08-06T14:30:00",
        "status": "AVAILABLE",
    }
]


def test_reschedule_after_confirmation_clears_old_date() -> None:
    state = {
        "messages": [
            HumanMessage(content="I want to reschedule"),
        ],
        "intent": "book_appointment",
        "appointment_id": 12,
        "appointment_reference_number": "APT-871E6728",
        "service_id": 2,
        "service_name": "Dental care",
        "staff_id": 5,
        "staff_name": "Dr. Perera",
        "requested_date": "2026-08-05",
        "available_slots": OLD_SLOTS,
        "slot_id": 7,
        "selected_slot_summary": "Old slot",
        "booking_summary": "Old booking",
        "confirmation_status": "confirmed",
        "available_services": DEMO_SERVICES,
    }

    updates = detect_intent(state)

    assert updates["intent"] == "reschedule_appointment"
    assert updates["requested_date"] is None
    assert updates["available_slots"] is None
    assert updates["slot_id"] is None
    assert updates["selected_slot_summary"] is None
    assert updates["booking_summary"] is None
    assert updates["confirmation_status"] == "not_requested"
    assert "appointment_id" not in updates
    assert "appointment_reference_number" not in updates

    response = determine_next_question(
        {
            **state,
            **updates,
        }
    )["next_question"]

    assert response == (
        "Which date would you like to move Dental care appointment to?"
    )


def test_reschedule_different_date_clears_selected_date() -> None:
    state = {
        "messages": [
            HumanMessage(content="no different date"),
        ],
        "intent": "reschedule_appointment",
        "appointment_id": 12,
        "service_id": 2,
        "service_name": "Dental care",
        "requested_date": "2026-08-05",
        "available_slots": OLD_SLOTS,
        "slot_id": 7,
        "selected_slot_summary": "Old slot",
        "booking_summary": "Old reschedule",
        "confirmation_status": "pending",
        "available_services": DEMO_SERVICES,
    }

    updates = detect_intent(state)

    assert updates["intent"] == "reschedule_appointment"
    assert updates["requested_date"] is None
    assert updates["available_slots"] is None
    assert updates["slot_id"] is None
    assert updates["selected_slot_summary"] is None
    assert updates["booking_summary"] is None
    assert updates["confirmation_status"] == "not_requested"

    response = determine_next_question(
        {
            **state,
            **updates,
        }
    )["next_question"]

    assert response == (
        "Which date would you like to move Dental care appointment to?"
    )


def test_booking_wording_with_date_stays_in_reschedule_flow() -> None:
    class FixedDate(date):
        @classmethod
        def today(cls) -> date:
            return cls(2026, 8, 4)

    state = {
        "messages": [
            HumanMessage(
                content="i want to book appointment on 06th",
            ),
        ],
        "intent": "reschedule_appointment",
        "appointment_id": 12,
        "service_id": 2,
        "service_name": "Dental care",
        "confirmation_status": "not_requested",
    }

    intent_updates = detect_intent(state)

    assert intent_updates["intent"] == "reschedule_appointment"

    with patch("app.ai.agent.date", FixedDate):
        detail_updates = extract_details(
            {
                **state,
                **intent_updates,
            }
        )

    assert detail_updates["requested_date"] == "2026-08-06"


def test_reschedule_reference_resolves_appointment_details() -> None:
    resolved_appointment = {
        "appointment_id": 12,
        "appointment_reference_number": "APT-871E6728",
        "service_id": 2,
        "service_name": "Dental care",
        "staff_id": 5,
        "staff_name": "Dr. Perera",
    }

    with patch(
        "app.ai.agent.get_appointment_by_reference",
        return_value=resolved_appointment,
    ) as mocked_lookup:
        updates = extract_details(
            {
                "messages": [
                    HumanMessage(content="APT-871E6728"),
                ],
                "intent": "reschedule_appointment",
            }
        )

    mocked_lookup.assert_called_once_with("APT-871E6728")
    assert updates == resolved_appointment


def test_human_handoff_exits_reschedule_question() -> None:
    state = {
        "messages": [
            HumanMessage(content="could you transfer to human"),
        ],
        "intent": "reschedule_appointment",
        "available_services": DEMO_SERVICES,
    }

    updates = detect_intent(state)
    response = determine_next_question(
        {
            **state,
            **updates,
        }
    )["next_question"]

    assert updates["intent"] == "general_question"
    assert response == (
        "I can hand this over to a human staff member. "
        "Please contact the front desk or clinic staff to continue "
        "this request."
    )
    assert "appointment ID" not in response


@pytest.mark.parametrize(
    "message",
    [
        "thank you",
        "sorry",
    ],
)
def test_polite_exit_stops_reschedule_question(message: str) -> None:
    state = {
        "messages": [
            HumanMessage(content=message),
        ],
        "intent": "reschedule_appointment",
        "available_services": DEMO_SERVICES,
    }

    updates = detect_intent(state)
    response = determine_next_question(
        {
            **state,
            **updates,
        }
    )["next_question"]

    assert updates["intent"] == "general_question"
    assert response is not None
    assert "appointment ID" not in response


def test_full_booking_then_reschedule_flow() -> None:
    thread_id = "full-reschedule-regression-001"
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    availability_tool = MagicMock()

    def available_slots(arguments: dict[str, object]):
        if arguments["requested_date"] == "2026-08-05":
            return OLD_SLOTS

        return NEW_SLOTS

    availability_tool.run.side_effect = available_slots

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
            availability_tool,
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
                "id": 12,
                "reference_number": "APT-871E6728",
                "start_datetime": "2026-08-05T10:00:00",
                "end_datetime": "2026-08-05T10:30:00",
            },
        ),
        patch(
            "app.ai.agent.reschedule_confirmed_appointment_from_state",
            return_value={
                "id": 12,
                "reference_number": "APT-871E6728",
                "start_datetime": "2026-08-06T14:00:00",
                "end_datetime": "2026-08-06T14:30:00",
            },
        ) as mocked_reschedule,
    ):
        messages = (
            "I need an appointment.",
            "Dental care.",
            "2026-08-05",
            "first one",
            "Kivaane Anton and contact number 0774588691",
            "yes",
        )

        result = {}

        for message in messages:
            result = appointment_agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=message),
                    ],
                },
                config=config,
            )

        assert result["appointment_id"] == 12
        assert result["confirmation_status"] == "confirmed"

        result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="I want to reschedule"),
                ],
            },
            config=config,
        )

        assert result["requested_date"] is None
        assert result["appointment_id"] == 12
        assert (
            "Which date would you like to move Dental care "
            "appointment to?"
        ) in extract_text_content(result["messages"][-1].content)

        result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="2026-08-06"),
                ],
            },
            config=config,
        )

        assert result["requested_date"] == "2026-08-06"
        assert result["available_slots"] == NEW_SLOTS

        result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="first one"),
                ],
            },
            config=config,
        )

        assert "Please confirm your rescheduled appointment" in (
            extract_text_content(result["messages"][-1].content)
        )

        result = appointment_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="yes"),
                ],
            },
            config=config,
        )

    mocked_reschedule.assert_called_once()
    assert "Your appointment has been rescheduled." in (
        extract_text_content(result["messages"][-1].content)
    )
