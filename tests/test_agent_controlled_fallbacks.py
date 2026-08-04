from unittest.mock import patch
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.ai.agent import (
    call_model,
    detect_intent,
    determine_next_question,
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


def controlled_response(message: str, **state_values):
    state = {
        "messages": [HumanMessage(content=message)],
        "available_services": SERVICES,
        **state_values,
    }
    intent_updates = detect_intent(state)
    question_updates = determine_next_question(
        {
            **state,
            **intent_updates,
        }
    )

    with patch("app.ai.agent.get_chat_model") as mocked_model:
        result = call_model(
            {
                **state,
                **intent_updates,
                **question_updates,
            }
        )

    mocked_model.assert_not_called()
    return intent_updates, question_updates, result


@pytest.mark.parametrize(
    "intent",
    [
        "book_appointment",
        "reschedule_appointment",
        "cancel_appointment",
        "check_availability",
        "view_appointments",
    ],
)
def test_handoff_interrupts_every_appointment_flow(intent: str) -> None:
    updates, question, _ = controlled_response(
        "talk to a person",
        intent=intent,
        appointment_id=12,
    )

    assert updates["intent"] == "general_question"
    assert question["next_question"] == (
        "I can hand this over to a human staff member. Please contact "
        "the front desk or clinic staff to continue this request."
    )


@pytest.mark.parametrize(
    "message",
    ["thank you", "thanks"],
)
def test_gratitude_interrupts_active_flow(message: str) -> None:
    _, question, _ = controlled_response(
        message,
        intent="book_appointment",
        service_id=2,
    )
    assert question["next_question"] == (
        "You're welcome. Let me know if you need help with anything else."
    )


@pytest.mark.parametrize(
    "message",
    ["never mind", "forget it", "stop"],
)
def test_abandon_phrases_stop_partial_flow(message: str) -> None:
    updates, question, _ = controlled_response(
        message,
        intent="reschedule_appointment",
        requested_date="2026-08-05",
        slot_id=7,
    )

    assert updates["requested_date"] is None
    assert updates["slot_id"] is None
    assert question["next_question"] == (
        "No problem, I've stopped that request. Let me know if you "
        "need help with anything else."
    )


def test_capabilities_request_is_controlled() -> None:
    _, question, _ = controlled_response("what can you do")
    assert "book, reschedule, or cancel" in str(question["next_question"])


def test_price_request_uses_service_data() -> None:
    _, question, _ = controlled_response("how much is dental")
    assert question["next_question"] == (
        "Dental care costs LKR 3,500 and takes 30 minutes."
    )


@pytest.mark.parametrize(
    "message",
    ["opening hours", "what is your cancellation policy"],
)
def test_unavailable_clinic_information_is_safe(message: str) -> None:
    _, question, _ = controlled_response(message)
    assert question["next_question"] == (
        "I don't have that information available yet. Please contact "
        "the front desk or clinic staff for accurate details."
    )


def test_how_are_you_is_controlled() -> None:
    _, question, _ = controlled_response("how are you")
    assert "ready to help with appointments" in str(
        question["next_question"],
    )


def test_blank_message_returns_controlled_prompt() -> None:
    result = determine_next_question(
        {
            "messages": [HumanMessage(content="   ")],
            "intent": "general_question",
            "available_services": SERVICES,
        }
    )
    assert result["next_question"] == (
        "Please type a message so I can help with your appointment."
    )


def test_gibberish_returns_controlled_fallback() -> None:
    _, question, _ = controlled_response("asdkjslkdj")
    assert "I didn't quite catch that" in str(question["next_question"])


@pytest.mark.parametrize(
    ("message", "expected_intent"),
    [
        ("I need an appoinmtent", "book_appointment"),
        ("I need to reshedule", "reschedule_appointment"),
        ("what slots are avalable", "check_availability"),
    ],
)
def test_common_typos_map_to_controlled_intents(
    message: str,
    expected_intent: str,
) -> None:
    result = detect_intent(
        {
            "messages": [HumanMessage(content=message)],
        }
    )
    assert result["intent"] == expected_intent


def test_provider_failure_returns_friendly_fallback() -> None:
    with patch(
        "app.ai.agent.get_chat_model",
        side_effect=RuntimeError("503 UNAVAILABLE"),
    ):
        result = call_model(
            {
                "messages": [HumanMessage(content="Tell me a story")],
                "next_question": None,
            }
        )

    assert result["messages"][0].content == (
        "I'm having trouble reaching the AI model right now, but I can "
        "still help with appointments. You can ask me to book, "
        "reschedule, cancel, check availability, or view your "
        "appointments."
    )


def test_gemini_model_disables_long_internal_retry_loop() -> None:
    settings = SimpleNamespace(
        ai_provider="gemini",
        google_api_key="test-key",
        gemini_model="gemini-test-model",
    )

    with (
        patch("app.ai.model.get_settings", return_value=settings),
        patch("app.ai.model.ChatGoogleGenerativeAI") as model_class,
    ):
        from app.ai.model import get_chat_model

        get_chat_model()

    model_class.assert_called_once_with(
        model="gemini-test-model",
        google_api_key="test-key",
        retries=0,
        request_timeout=10,
    )
