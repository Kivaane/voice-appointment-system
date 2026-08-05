from fastapi.testclient import TestClient

from app.main import app
from app.routes import ai_chat


client = TestClient(app)


def test_ai_chat_uses_provided_thread_id(
    monkeypatch,
) -> None:
    captured_arguments: dict[str, str] = {}

    def fake_run_appointment_agent(
        user_message: str,
        thread_id: str,
        request_id: str,
    ) -> str:
        captured_arguments["user_message"] = user_message
        captured_arguments["thread_id"] = thread_id
        captured_arguments["request_id"] = request_id

        return "How may I help with your appointment?"

    monkeypatch.setattr(
        ai_chat,
        "run_appointment_agent",
        fake_run_appointment_agent,
    )

    response = client.post(
        "/ai/chat",
        json={
            "message": "I need an appointment.",
            "thread_id": "thread-chat-test-001",
        },
    )

    assert response.status_code == 200

    response_data = response.json()
    assert response_data["thread_id"] == "thread-chat-test-001"
    assert response_data["response"] == (
        "How may I help with your appointment?"
    )
    assert response_data["message"] == response_data["response"]
    assert response_data["conversation_stage"] == "idle"

    assert captured_arguments["user_message"] == (
        "I need an appointment."
    )

    assert captured_arguments["thread_id"] == (
        "thread-chat-test-001"
    )

    assert captured_arguments["request_id"]


def test_ai_chat_generates_thread_id(
    monkeypatch,
) -> None:
    captured_thread_id: dict[str, str] = {}

    def fake_run_appointment_agent(
        user_message: str,
        thread_id: str,
        request_id: str,
    ) -> str:
        captured_thread_id["value"] = thread_id

        return "Please tell me which service you need."

    monkeypatch.setattr(
        ai_chat,
        "run_appointment_agent",
        fake_run_appointment_agent,
    )

    response = client.post(
        "/ai/chat",
        json={
            "message": "Hello",
        },
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["thread_id"]
    assert response_data["thread_id"] == (
        captured_thread_id["value"]
    )

    assert response_data["response"] == (
        "Please tell me which service you need."
    )


def test_ai_chat_rejects_empty_message() -> None:
    response = client.post(
        "/ai/chat",
        json={
            "message": "",
        },
    )

    assert response.status_code == 422
