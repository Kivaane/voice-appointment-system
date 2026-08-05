from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from app.ai.agent import (
    determine_next_question,
    extract_details,
    extract_time_preference,
    filter_slots_by_time_preference,
    lookup_conversation_availability,
    resolve_named_entities,
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


SLOTS = [
    {
        "slot_id": 7,
        "service_id": 2,
        "service_name": "Dental care",
        "staff_id": 5,
        "staff_name": "Dr. Perera",
        "start_datetime": "2026-08-06T09:00:00",
        "end_datetime": "2026-08-06T09:30:00",
    },
    {
        "slot_id": 8,
        "service_id": 2,
        "service_name": "Dental care",
        "staff_id": 5,
        "staff_name": "Dr. Perera",
        "start_datetime": "2026-08-06T14:00:00",
        "end_datetime": "2026-08-06T14:30:00",
    },
]


def test_service_and_named_date_are_extracted_in_one_message() -> None:
    state = {
        "messages": [HumanMessage(content="Book dental on 6 August")],
        "intent": "book_appointment",
    }
    extracted = extract_details(state)

    with (
        patch("app.ai.agent.get_active_services", return_value=SERVICES),
        patch(
            "app.ai.agent.get_active_staff_for_service",
            return_value=[],
        ),
    ):
        resolved = resolve_named_entities({**state, **extracted})

    assert extracted["requested_date"] == "2026-08-06"
    assert resolved["service_id"] == 2


def test_full_message_extracts_name_and_normalized_phone() -> None:
    result = extract_details(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Book physiotherapy next Monday for Kivaane "
                        "Anton, phone 0774588691"
                    )
                )
            ],
            "intent": "book_appointment",
        }
    )

    assert result["customer_name"] == "Kivaane Anton"
    assert result["customer_phone_number"] == "+94774588691"
    assert result["requested_date"] is not None


def test_time_period_filters_slots_before_selection() -> None:
    preference, error = extract_time_preference(
        "I need dental on 6 August in the morning"
    )

    assert error is None
    assert preference is not None
    assert filter_slots_by_time_preference(SLOTS, preference) == [SLOTS[0]]


def test_before_and_after_preferences_filter_slots() -> None:
    after, _ = extract_time_preference("after 1 PM")
    before, _ = extract_time_preference("before 11 AM")

    assert filter_slots_by_time_preference(SLOTS, after) == [SLOTS[1]]
    assert filter_slots_by_time_preference(SLOTS, before) == [SLOTS[0]]


def test_invalid_phone_in_full_message_uses_controlled_validation() -> None:
    state = {
        "messages": [
            HumanMessage(
                content=(
                    "Book dental tomorrow. My name is Kivaane Anton "
                    "and my number is 07756688791"
                )
            )
        ],
        "intent": "book_appointment",
        "available_services": SERVICES,
    }
    extracted = extract_details(state)
    response = determine_next_question({**state, **extracted})

    assert extracted["customer_name"] == "Kivaane Anton"
    assert extracted["customer_phone_invalid"] is True
    assert "valid Sri Lankan phone number" in response["next_question"]


def test_unknown_service_is_not_invented() -> None:
    state = {
        "messages": [HumanMessage(content="Book surgery tomorrow")],
        "intent": "book_appointment",
    }

    with (
        patch("app.ai.agent.get_active_services", return_value=SERVICES),
        patch(
            "app.ai.agent.get_active_staff_for_service",
            return_value=[],
        ),
    ):
        result = resolve_named_entities(state)

    assert result.get("service_id") is None


def test_ambiguous_time_produces_precise_clarification() -> None:
    state = {
        "messages": [
            HumanMessage(content="Book dental tomorrow around 2")
        ],
        "intent": "book_appointment",
        "available_services": SERVICES,
    }
    extracted = extract_details(state)
    response = determine_next_question({**state, **extracted})

    assert extracted["time_preference_error"] is not None
    assert "clarify the time" in response["next_question"]


def test_single_matching_preference_selects_slot_without_booking() -> None:
    mocked_tool = MagicMock()
    mocked_tool.run.return_value = SLOTS
    preference, _ = extract_time_preference("in the morning")

    with patch("app.ai.agent.check_available_slots", mocked_tool):
        result = lookup_conversation_availability(
            {
                "messages": [HumanMessage(content="in the morning")],
                "intent": "book_appointment",
                "service_id": 2,
                "staff_id": 5,
                "requested_date": "2026-08-06",
                "time_preference": preference,
            }
        )

    assert result["slot_id"] == 7
    assert result.get("appointment_id") is None
