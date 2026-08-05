from langchain_core.messages import HumanMessage

from app.ai import agent


SERVICES = [
    {"id": 1, "name": "Dental care"},
    {"id": 2, "name": "Physiotherapy"},
]
STAFF = [
    {"id": 5, "full_name": "Dr. Perera"},
    {"id": 6, "full_name": "Dr. Silva"},
]


def booking_state(message: str) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "intent": "book_appointment",
        "service_id": 1,
        "service_name": "Dental care",
        "staff_id": 5,
        "staff_name": "Dr. Perera",
        "requested_date": "2026-08-06",
        "available_slots": [{"slot_id": 10}],
        "slot_id": 10,
        "selected_slot_summary": "Dental at 10:00 AM",
        "customer_id": 9,
        "customer_name": "Old Name",
        "customer_phone_number": "+94774588691",
        "booking_summary": "Confirm booking",
        "confirmation_status": "pending",
    }


def test_natural_date_correction_clears_only_date_dependencies() -> None:
    state = booking_state("No, I meant Friday")
    intent_updates = agent.detect_intent(state)
    extracted = agent.extract_details({**state, **intent_updates})
    result = {**state, **intent_updates, **extracted}

    assert result["service_id"] == 1
    assert result["customer_id"] == 9
    assert result["requested_date"] != "2026-08-06"
    assert result["slot_id"] is None
    assert result["booking_summary"] is None
    assert result["confirmation_status"] == "not_requested"


def test_natural_time_correction_preserves_date_and_customer() -> None:
    state = booking_state("Afternoon instead")
    intent_updates = agent.detect_intent(state)
    extracted = agent.extract_details({**state, **intent_updates})
    result = {**state, **intent_updates, **extracted}

    assert result["requested_date"] == "2026-08-06"
    assert result["service_id"] == 1
    assert result["customer_id"] == 9
    assert result["slot_id"] is None
    assert result["time_preference"]["period"] == "afternoon"
    assert result["confirmation_status"] == "not_requested"


def test_targeted_service_correction_preserves_date_and_customer(
    monkeypatch,
) -> None:
    state = booking_state("Not dental, physiotherapy")
    monkeypatch.setattr(agent, "get_active_services", lambda: SERVICES)
    monkeypatch.setattr(
        agent,
        "get_active_staff_for_service",
        lambda service_id: STAFF,
    )
    intent_updates = agent.detect_intent(state)
    result = agent.resolve_named_entities({**state, **intent_updates})

    assert result["service_id"] == 2
    assert result["service_name"] == "Physiotherapy"
    assert state["requested_date"] == "2026-08-06"
    assert state["customer_id"] == 9
    assert result["slot_id"] is None
    assert result["staff_id"] is None
    assert result["confirmation_status"] == "not_requested"


def test_staff_correction_clears_only_staff_availability(monkeypatch) -> None:
    state = booking_state("Change only the doctor to Dr. Silva")
    monkeypatch.setattr(agent, "get_active_services", lambda: SERVICES)
    monkeypatch.setattr(
        agent,
        "get_active_staff_for_service",
        lambda service_id: STAFF,
    )
    intent_updates = agent.detect_intent(state)
    resolved = agent.resolve_named_entities({**state, **intent_updates})
    result = {**state, **intent_updates, **resolved}

    assert result["staff_id"] == 6
    assert result["requested_date"] == "2026-08-06"
    assert result["service_id"] == 1
    assert result["customer_id"] == 9
    assert result["slot_id"] is None


def test_phone_correction_preserves_selected_appointment_details() -> None:
    state = booking_state("Use my other number: 0771234567")
    result = agent.extract_details(state)

    assert result["customer_phone_number"] == "+94771234567"
    assert result["customer_id"] is None
    assert result["booking_summary"] is None
    assert state["service_id"] == 1
    assert state["slot_id"] == 10


def test_name_correction_preserves_selected_appointment_details() -> None:
    state = booking_state("Sorry, my name is Kivaane Anton")
    result = agent.extract_details(state)

    assert result["customer_name"] == "Kivaane Anton"
    assert result["customer_id"] is None
    assert result["booking_summary"] is None
    assert state["service_id"] == 1
    assert state["slot_id"] == 10
