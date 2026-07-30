from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.ai_persistence import (
    get_or_create_conversation,
    list_conversation_events,
    list_conversation_messages,
    record_event,
    save_message,
)
from app.database import Base
from app.models import AIEventType, AIMessageRole


def test_ai_persistence_lifecycle() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    Base.metadata.create_all(bind=engine)

    with Session(engine) as database:
        conversation = get_or_create_conversation(
            database=database,
            thread_id="thread-test-001",
            current_intent="book_appointment",
        )

        same_conversation = get_or_create_conversation(
            database=database,
            thread_id="thread-test-001",
        )

        assert conversation.id == same_conversation.id
        assert conversation.thread_id == "thread-test-001"

        user_message = save_message(
            database=database,
            conversation_id=conversation.id,
            role=AIMessageRole.USER,
            content="I need a dental appointment.",
        )

        assistant_message = save_message(
            database=database,
            conversation_id=conversation.id,
            role=AIMessageRole.ASSISTANT,
            content="I will check available services.",
            model_name="gemini-3.5-flash-lite",
        )

        event = record_event(
            database=database,
            conversation_id=conversation.id,
            event_type=AIEventType.TOOL_CALL,
            event_name="list_available_services",
            event_data={
                "success": True,
                "result_count": 3,
            },
            request_id="request-test-001",
        )

        messages = list_conversation_messages(
            database=database,
            conversation_id=conversation.id,
        )

        events = list_conversation_events(
            database=database,
            conversation_id=conversation.id,
        )

        assert [message.id for message in messages] == [
            user_message.id,
            assistant_message.id,
        ]

        assert messages[0].role == AIMessageRole.USER
        assert messages[1].role == AIMessageRole.ASSISTANT
        assert messages[1].model_name == "gemini-3.5-flash-lite"

        assert len(events) == 1
        assert events[0].id == event.id
        assert events[0].event_type == AIEventType.TOOL_CALL
        assert events[0].event_data == {
            "success": True,
            "result_count": 3,
        }
        assert events[0].request_id == "request-test-001"