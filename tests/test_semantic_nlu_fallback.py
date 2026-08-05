import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from app.ai import agent
from app.ai.nlu import NLUModelResult


@pytest.mark.parametrize(
    ("message", "semantic_intent"),
    [
        ("Can you shift my booking?", "reschedule_appointment"),
        ("I won't be able to make it.", "cancel_appointment"),
        ("Get me the earliest tooth appointment.", "book_appointment"),
        ("What will dentistry set me back?", "ask_pricing"),
        ("When's my thing with the doctor?", "view_appointments"),
    ],
)
def test_unknown_paraphrases_use_validated_semantic_classification(
    monkeypatch,
    message: str,
    semantic_intent: str,
) -> None:
    monkeypatch.setattr(
        agent,
        "classify_unknown_message",
        lambda user_message: NLUModelResult(
            intent=semantic_intent,
            confidence=0.88,
        ),
    )

    result = agent.detect_intent(
        {
            "messages": [HumanMessage(content=message)],
            "intent": None,
            "confirmation_status": "not_requested",
        }
    )

    expected_intent = (
        "general_question"
        if semantic_intent == "ask_pricing"
        else semantic_intent
    )
    assert result["intent"] == expected_intent
    assert result["semantic_nlu"]["confidence"] == 0.88


def test_semantic_hints_feed_deterministic_entity_resolution(monkeypatch) -> None:
    monkeypatch.setattr(
        agent,
        "classify_unknown_message",
        lambda user_message: NLUModelResult(
            intent="book_appointment",
            confidence=0.9,
            service_hint="Dental care",
            date_hint="tomorrow",
            time_hint="morning",
        ),
    )
    monkeypatch.setattr(
        agent,
        "get_active_services",
        lambda: [{"id": 2, "name": "Dental care"}],
    )
    monkeypatch.setattr(
        agent,
        "get_active_staff_for_service",
        lambda service_id: [],
    )
    state = {
        "messages": [
            HumanMessage(content="Get me the earliest tooth appointment")
        ],
        "intent": None,
        "confirmation_status": "not_requested",
    }
    intent_updates = agent.detect_intent(state)
    extracted = agent.extract_details({**state, **intent_updates})
    resolved = agent.resolve_named_entities(
        {**state, **intent_updates, **extracted}
    )

    assert resolved["service_id"] == 2
    assert extracted["requested_date"] is not None
    assert extracted["time_preference"]["period"] == "morning"


def test_known_message_never_calls_semantic_model(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        agent,
        "classify_unknown_message",
        lambda user_message: calls.append(user_message),
    )

    result = agent.detect_intent(
        {
            "messages": [HumanMessage(content="Book dental tomorrow")],
            "intent": None,
        }
    )

    assert result["intent"] == "book_appointment"
    assert calls == []


def test_pending_confirmation_never_calls_semantic_model(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        agent,
        "classify_unknown_message",
        lambda user_message: calls.append(user_message),
    )

    agent.detect_intent(
        {
            "messages": [HumanMessage(content="perhaps do it")],
            "intent": "book_appointment",
            "confirmation_status": "pending",
            "booking_summary": "Confirm",
        }
    )

    assert calls == []


def test_human_handoff_never_calls_semantic_model(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        agent,
        "classify_unknown_message",
        lambda user_message: calls.append(user_message),
    )

    result = agent.detect_intent(
        {
            "messages": [HumanMessage(content="I want to speak to a person")],
            "intent": "book_appointment",
            "service_id": 2,
        }
    )

    assert calls == []
    assert result["intent"] == "general_question"
    assert "human staff member" in result["next_question"]


def test_invalid_semantic_schema_is_rejected() -> None:
    with pytest.raises(ValidationError):
        NLUModelResult.model_validate(
            {"intent": "delete_database", "confidence": 1.0}
        )
