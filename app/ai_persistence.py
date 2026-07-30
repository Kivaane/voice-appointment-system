from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    AIConversation,
    AIConversationStatus,
    AIEvent,
    AIEventType,
    AIMessage,
    AIMessageRole,
)


def get_conversation_by_thread_id(
    database: Session,
    thread_id: str,
) -> AIConversation | None:
    """Find an AI conversation using its LangGraph thread ID."""

    return database.scalar(
        select(AIConversation).where(
            AIConversation.thread_id == thread_id
        )
    )


def get_or_create_conversation(
    database: Session,
    thread_id: str,
    customer_id: int | None = None,
    current_intent: str | None = None,
) -> AIConversation:
    """Return an existing conversation or create a new one."""

    conversation = get_conversation_by_thread_id(
        database,
        thread_id,
    )

    if conversation is not None:
        return conversation

    conversation = AIConversation(
        thread_id=thread_id,
        customer_id=customer_id,
        current_intent=current_intent,
        status=AIConversationStatus.ACTIVE,
    )

    database.add(conversation)
    database.commit()
    database.refresh(conversation)

    return conversation


def save_message(
    database: Session,
    conversation_id: int,
    role: AIMessageRole,
    content: str,
    model_name: str | None = None,
) -> AIMessage:
    """Save one message belonging to an AI conversation."""

    message = AIMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
        model_name=model_name,
    )

    database.add(message)
    database.commit()
    database.refresh(message)

    return message


def record_event(
    database: Session,
    conversation_id: int,
    event_type: AIEventType,
    event_name: str | None = None,
    event_data: dict | list | None = None,
    request_id: str | None = None,
) -> AIEvent:
    """Record a structured event from an AI agent run."""

    event = AIEvent(
        conversation_id=conversation_id,
        request_id=request_id,
        event_type=event_type,
        event_name=event_name,
        event_data=event_data,
    )

    database.add(event)
    database.commit()
    database.refresh(event)

    return event


def list_conversation_messages(
    database: Session,
    conversation_id: int,
) -> list[AIMessage]:
    """Return conversation messages in their original order."""

    return list(
        database.scalars(
            select(AIMessage)
            .where(
                AIMessage.conversation_id == conversation_id
            )
            .order_by(
                AIMessage.created_at,
                AIMessage.id,
            )
        )
    )


def list_conversation_events(
    database: Session,
    conversation_id: int,
) -> list[AIEvent]:
    """Return recorded AI events in their original order."""

    return list(
        database.scalars(
            select(AIEvent)
            .where(
                AIEvent.conversation_id == conversation_id
            )
            .order_by(
                AIEvent.created_at,
                AIEvent.id,
            )
        )
    )
