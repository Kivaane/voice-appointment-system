from langchain_core.messages import HumanMessage

from app.ai import agent


def state_for(message: str) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "intent": None,
        "confirmation_status": "not_requested",
    }


def test_information_and_availability_choose_safe_primary_goal() -> None:
    result = agent.detect_intent(
        state_for("How much is dental, and is it available tomorrow?")
    )

    assert result["intent"] == "check_availability"
    assert result["secondary_intents"] == ["ask_pricing"]


def test_status_and_cancellation_choose_cancellation_without_mutating() -> None:
    result = agent.detect_intent(
        state_for("Show my appointments and cancel the dental one")
    )

    assert result["intent"] == "cancel_appointment"
    assert "view_appointments" in result["secondary_intents"]
    assert result["confirmation_status"] == "not_requested"


def test_cancellation_and_booking_require_action_order_clarification() -> None:
    result = agent.detect_intent(
        state_for("Cancel my old appointment and book new appointment")
    )

    assert result["intent"] == "general_question"
    assert "one change at a time" in result["next_question"]
    assert result["booking_summary"] is None
    assert result["confirmation_status"] == "not_requested"


def test_policy_then_reschedule_preserves_reschedule_as_primary() -> None:
    result = agent.detect_intent(
        state_for(
            "Move my appointment, but first tell me the cancellation policy"
        )
    )

    assert result["intent"] == "reschedule_appointment"
    assert result["secondary_intents"] == ["ask_cancellation_policy"]


def test_conflicting_mutations_never_execute_action(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        agent,
        "create_confirmed_appointment_from_state",
        lambda state: calls.append("book"),
    )
    monkeypatch.setattr(
        agent,
        "cancel_confirmed_appointment_from_state",
        lambda state: calls.append("cancel"),
    )
    updates = agent.detect_intent(
        state_for("Cancel my appointment and book new appointment")
    )
    action_result = agent.confirm_or_reject_booking(updates)

    assert calls == []
    assert action_result == {}

