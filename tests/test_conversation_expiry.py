from datetime import datetime, timedelta, timezone

from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.ai import agent
from app.ai_persistence import (
    get_or_create_conversation,
    load_conversation_state,
    save_conversation_state,
)
from app.database import Base


NOW = datetime(2026, 8, 4, 10, 0, tzinfo=timezone.utc)


def old_timestamp(minutes: int = 60) -> str:
    return (NOW - timedelta(minutes=minutes)).isoformat()


def test_fresh_transaction_state_continues_normally() -> None:
    state = {
        "intent": "book_appointment",
        "service_id": 2,
        "requested_date": "2026-08-06",
        "confirmation_status": "not_requested",
        "transaction_updated_at": (NOW - timedelta(minutes=5)).isoformat(),
    }

    recovered = agent.expire_stale_transaction_state(state, now=NOW)

    assert recovered["intent"] == "book_appointment"
    assert recovered["service_id"] == 2
    assert recovered["state_expired_message"] is None


def test_expired_booking_state_is_safely_reset() -> None:
    recovered = agent.expire_stale_transaction_state(
        {
            "intent": "book_appointment",
            "customer_id": 9,
            "customer_name": "Kivaane Anton",
            "service_id": 2,
            "requested_date": "2026-08-06",
            "slot_id": 12,
            "confirmation_status": "not_requested",
            "transaction_updated_at": old_timestamp(),
        },
        now=NOW,
    )

    assert recovered["intent"] == "general_question"
    assert recovered["service_id"] is None
    assert recovered["slot_id"] is None
    assert recovered["customer_id"] == 9
    assert "expired for safety" in recovered["state_expired_message"]


def test_expired_pending_confirmation_cannot_execute(monkeypatch) -> None:
    mutation_calls = []
    monkeypatch.setattr(
        agent,
        "create_confirmed_appointment_from_state",
        lambda state: mutation_calls.append(state),
    )
    recovered = agent.expire_stale_transaction_state(
        {
            "intent": "book_appointment",
            "confirmation_status": "pending",
            "booking_summary": "Confirm this booking?",
            "customer_id": 1,
            "service_id": 2,
            "staff_id": 3,
            "slot_id": 4,
            "pending_action_started_at": old_timestamp(),
            "transaction_updated_at": old_timestamp(5),
        },
        now=NOW,
    )

    updates = agent.detect_intent(
        {**recovered, "messages": [HumanMessage(content="yes")]}
    )
    result = agent.confirm_or_reject_booking({**recovered, **updates})

    assert recovered["confirmation_status"] == "not_requested"
    assert recovered["booking_summary"] is None
    assert mutation_calls == []
    assert result == {}


def test_expired_slot_options_are_rechecked(monkeypatch) -> None:
    calls = []
    current_slots = [
        {
            "slot_id": 18,
            "service_id": 2,
            "staff_id": 3,
            "staff_name": "Dr. Perera",
            "start_datetime": "2026-08-06T09:00:00",
            "end_datetime": "2026-08-06T09:30:00",
            "status": "AVAILABLE",
        }
    ]
    monkeypatch.setattr(
        agent,
        "get_validated_available_slots",
        lambda arguments: calls.append(arguments) or current_slots,
    )
    recovered = agent.expire_stale_transaction_state(
        {
            "intent": "book_appointment",
            "service_id": 2,
            "requested_date": "2026-08-06",
            "slot_id": 7,
            "available_slots": [{"slot_id": 7}],
            "confirmation_status": "not_requested",
            "transaction_updated_at": old_timestamp(5),
            "slot_options_updated_at": old_timestamp(),
            "messages": [HumanMessage(content="yes")],
        },
        now=NOW,
    )

    result = agent.lookup_conversation_availability(recovered)

    assert recovered["slot_id"] is None
    assert calls == [
        {"service_id": 2, "requested_date": "2026-08-06"}
    ]
    assert result["available_slots"] == current_slots


def test_completed_conversation_is_not_reopened() -> None:
    state = {
        "intent": "book_appointment",
        "appointment_id": 77,
        "confirmation_status": "confirmed",
        "transaction_updated_at": old_timestamp(),
    }

    recovered = agent.expire_stale_transaction_state(state, now=NOW)

    assert recovered["intent"] == "book_appointment"
    assert recovered["appointment_id"] == 77
    assert recovered["confirmation_status"] == "confirmed"
    assert recovered["state_expired_message"] is None


def test_expiry_does_not_mix_threads() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)

    with Session(engine) as database:
        expired = get_or_create_conversation(database, "expired-thread")
        fresh = get_or_create_conversation(database, "fresh-thread")
        save_conversation_state(
            database,
            expired,
            {
                "intent": "book_appointment",
                "service_id": 2,
                "transaction_updated_at": old_timestamp(),
                "confirmation_status": "not_requested",
            },
            "book_appointment",
        )
        save_conversation_state(
            database,
            fresh,
            {
                "intent": "reschedule_appointment",
                "appointment_id": 8,
                "transaction_updated_at": old_timestamp(5),
                "confirmation_status": "not_requested",
            },
            "reschedule_appointment",
        )

        expired_state = agent.expire_stale_transaction_state(
            load_conversation_state(expired),
            now=NOW,
        )
        fresh_state = load_conversation_state(fresh)

    assert expired_state["intent"] == "general_question"
    assert fresh_state["intent"] == "reschedule_appointment"
    assert fresh_state["appointment_id"] == 8
