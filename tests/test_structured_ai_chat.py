from fastapi.testclient import TestClient

from app.main import app
from app.routes import ai_chat


def test_normal_reply_is_backward_compatible_and_structured(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ai_chat,
        "run_appointment_agent",
        lambda **kwargs: "How can I help?",
    )
    monkeypatch.setattr(
        ai_chat,
        "get_structured_conversation_state",
        lambda thread_id: {
            "intent": "general_question",
            "conversation_stage": "idle",
            "requires_confirmation": False,
            "pending_action": None,
            "options": [],
            "error": None,
        },
    )

    with TestClient(app) as client:
        response = client.post(
            "/ai/chat",
            json={
                "message": "Hello",
                "thread_id": "structured-thread",
                "request_id": "structured-request",
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["thread_id"] == "structured-thread"
    assert body["response"] == "How can I help?"
    assert body["message"] == body["response"]
    assert body["conversation_stage"] == "idle"


def test_slot_and_confirmation_metadata_schema(monkeypatch) -> None:
    monkeypatch.setattr(
        ai_chat,
        "run_appointment_agent",
        lambda **kwargs: "Please confirm your appointment.",
    )
    monkeypatch.setattr(
        ai_chat,
        "get_structured_conversation_state",
        lambda thread_id: {
            "intent": "book_appointment",
            "conversation_stage": "awaiting_confirmation",
            "requires_confirmation": True,
            "pending_action": "book_appointment",
            "options": [
                {
                    "id": 12,
                    "label": "10:00 AM with Dr. Perera",
                    "start_datetime": "2026-08-06T10:00:00",
                    "end_datetime": "2026-08-06T10:30:00",
                }
            ],
            "error": None,
        },
    )

    with TestClient(app) as client:
        response = client.post(
            "/ai/chat",
            json={"message": "option 1", "thread_id": "slot-thread"},
        )

    body = response.json()
    assert body["requires_confirmation"] is True
    assert body["pending_action"] == "book_appointment"
    assert body["options"][0]["id"] == 12


def test_controlled_error_metadata_schema(monkeypatch) -> None:
    monkeypatch.setattr(
        ai_chat,
        "run_appointment_agent",
        lambda **kwargs: "I couldn't verify availability right now.",
    )
    monkeypatch.setattr(
        ai_chat,
        "get_structured_conversation_state",
        lambda thread_id: {
            "intent": "check_availability",
            "conversation_stage": "checking_availability",
            "requires_confirmation": False,
            "pending_action": None,
            "options": [],
            "error": "Availability temporarily unavailable.",
        },
    )

    with TestClient(app) as client:
        response = client.post(
            "/ai/chat",
            json={"message": "Dental tomorrow"},
        )

    assert response.status_code == 200
    assert response.json()["error"] == "Availability temporarily unavailable."
