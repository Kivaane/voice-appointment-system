from datetime import datetime, timedelta, timezone

from langchain_core.messages import AIMessage
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.ai import agent
from app.ai_persistence import (
    begin_request_execution,
    get_or_create_conversation,
)
from app.database import Base
from app.models import AIMessage as DatabaseAIMessage, AIRequestExecution


def test_same_request_id_returns_completed_response_once(monkeypatch) -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(agent, "SessionLocal", session_local)

    class FakeModel:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            return AIMessage(content="One durable response.")

    model = FakeModel()
    monkeypatch.setattr(agent, "get_chat_model", lambda: model)

    first = agent.run_appointment_agent(
        "Tell me something unrelated.",
        thread_id="idempotency-thread",
        request_id="same-request",
    )
    second = agent.run_appointment_agent(
        "Tell me something unrelated.",
        thread_id="idempotency-thread",
        request_id="same-request",
    )

    assert first == "One durable response."
    assert second == first
    assert model.calls == 1

    with session_local() as database:
        message_count = database.scalar(
            select(func.count(DatabaseAIMessage.id))
        )
        executions = list(database.scalars(select(AIRequestExecution)))
        assert message_count == 2
        assert len(executions) == 1
        assert executions[0].status == "completed"


def test_repeated_completed_confirmation_does_not_execute_again(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        agent,
        "create_confirmed_appointment_from_state",
        lambda state: calls.append(state) or {
            "id": 44,
            "reference_number": "APT-ONCE",
            "slot_id": 7,
            "start_datetime": "2026-08-06T09:00:00",
            "end_datetime": "2026-08-06T09:30:00",
        },
    )
    pending_state = {
        "intent": "book_appointment",
        "confirmation_status": "confirmed",
        "booking_summary": "Please confirm",
        "customer_id": 1,
        "service_id": 2,
        "staff_id": 5,
        "slot_id": 7,
        "service_name": "Dental care",
        "staff_name": "Dr. Perera",
        "requested_date": "2026-08-06",
    }

    first = agent.confirm_or_reject_booking(pending_state)
    second = agent.confirm_or_reject_booking(
        {
            **pending_state,
            **first,
            "booking_summary": None,
        }
    )

    assert len(calls) == 1
    assert second == {}


def test_repeated_cancellation_confirmation_cancels_once(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        agent,
        "cancel_confirmed_appointment_from_state",
        lambda state: calls.append(state) or {
            "appointment_reference_number": "APT-CANCEL-ONCE",
            "appointment_status": "CANCELLED_BY_CUSTOMER",
        },
    )
    pending_state = {
        "intent": "cancel_appointment",
        "confirmation_status": "confirmed",
        "booking_summary": "Confirm cancellation",
        "appointment_id": 31,
    }

    first = agent.confirm_or_reject_booking(pending_state)
    second = agent.confirm_or_reject_booking({**pending_state, **first})

    assert len(calls) == 1
    assert second == {}


def test_repeated_reschedule_confirmation_reschedules_once(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        agent,
        "reschedule_confirmed_appointment_from_state",
        lambda state: calls.append(state) or {
            "id": 51,
            "reference_number": "APT-MOVE-ONCE",
            "slot_id": 9,
            "start_datetime": "2026-08-07T10:00:00",
            "end_datetime": "2026-08-07T10:30:00",
        },
    )
    pending_state = {
        "intent": "reschedule_appointment",
        "confirmation_status": "confirmed",
        "booking_summary": "Confirm reschedule",
        "appointment_id": 51,
        "current_slot_id": 7,
        "slot_id": 9,
        "service_name": "Dental care",
        "staff_name": "Dr. Perera",
        "requested_date": "2026-08-07",
    }

    first = agent.confirm_or_reject_booking(pending_state)
    second = agent.confirm_or_reject_booking({**pending_state, **first})

    assert len(calls) == 1
    assert "already your current appointment time" in second["next_question"]


def test_stale_executing_request_can_be_retried() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)

    with session_local() as database:
        conversation = get_or_create_conversation(database, "timeout-thread")
        first, cached = begin_request_execution(
            database,
            conversation.id,
            "timeout-request",
        )
        first.updated_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        database.commit()

        retried, cached = begin_request_execution(
            database,
            conversation.id,
            "timeout-request",
        )

        assert cached is None
        assert retried.id == first.id
        assert retried.status == "executing"
        retried_at = retried.updated_at
        if retried_at.tzinfo is None:
            retried_at = retried_at.replace(tzinfo=timezone.utc)
        assert retried_at > datetime.now(timezone.utc) - timedelta(minutes=1)
