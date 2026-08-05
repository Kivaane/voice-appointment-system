from langchain_core.messages import HumanMessage

from app.ai import agent


def test_competing_weekdays_receive_concrete_date_choices() -> None:
    state = {
        "messages": [HumanMessage(content="Friday or Saturday")],
        "intent": "book_appointment",
        "service_id": 1,
        "confirmation_status": "not_requested",
    }
    extracted = agent.extract_details(state)
    response = agent.determine_next_question({**state, **extracted})

    assert extracted["requested_date"] is None
    assert "Do you mean" in response["next_question"]
    assert "Friday" in response["next_question"]
    assert "Saturday" in response["next_question"]


def test_third_failed_slot_clarification_offers_handoff() -> None:
    response = agent.determine_next_question(
        {
            "messages": [HumanMessage(content="this one")],
            "intent": "book_appointment",
            "available_slots": [
                {
                    "slot_id": 1,
                    "staff_id": 2,
                    "start_datetime": "2026-08-06T10:00:00",
                    "end_datetime": "2026-08-06T10:30:00",
                },
                {
                    "slot_id": 2,
                    "staff_id": 2,
                    "start_datetime": "2026-08-06T14:30:00",
                    "end_datetime": "2026-08-06T15:00:00",
                },
            ],
            "slot_selection_error": "Which slot do you mean?",
            "clarification_attempts": 2,
        }
    )

    assert response["clarification_attempts"] == 3
    assert "front desk" in response["next_question"]

