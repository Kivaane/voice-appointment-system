from langchain_core.messages import HumanMessage

from app.ai.agent import determine_next_question, extract_details


SLOTS = [
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


def slot_state(message: str, slots=None) -> dict[str, object]:
    return {
        "messages": [HumanMessage(content=message)],
        "intent": "book_appointment",
        "service_id": 2,
        "service_name": "Dental care",
        "requested_date": "2026-08-05",
        "available_slots": slots or SLOTS,
        "confirmation_status": "not_requested",
    }


def test_single_slot_oki_this_slot_selects_it() -> None:
    updates = extract_details(slot_state("oki this slot", [SLOTS[0]]))
    assert updates["slot_id"] == 7


def test_single_slot_this_one_selects_it() -> None:
    updates = extract_details(slot_state("this one", [SLOTS[0]]))
    assert updates["slot_id"] == 7


def test_multiple_slots_this_one_asks_clarification() -> None:
    state = slot_state("this one")
    updates = extract_details(state)

    response = determine_next_question(
        {
            **state,
            **updates,
        }
    )["next_question"]

    assert response == (
        "Which slot do you mean? Please choose option 1, 2, or a time."
    )


def test_option_two_selects_second_displayed_slot() -> None:
    updates = extract_details(slot_state("option 2"))
    assert updates["slot_id"] == 8


def test_slot_one_uses_display_option_before_database_id() -> None:
    updates = extract_details(slot_state("slot 1"))
    assert updates["slot_id"] == 7


def test_invalid_slot_option_returns_controlled_response() -> None:
    state = slot_state("slot 5")
    updates = extract_details(state)
    response = determine_next_question(
        {
            **state,
            **updates,
        }
    )["next_question"]

    assert response == (
        "I only have options 1 and 2 available. Would you like one "
        "of those, or a different date?"
    )


def test_phone_number_is_not_parsed_as_slot_choice() -> None:
    updates = extract_details(slot_state("0771234567"))
    assert updates.get("slot_id") is None
