from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import (
    AIConversation,
    AIConversationStatus,
    AIEvent,
    AIEventType,
    AIMessage,
    AIMessageRole,
    AIRequestExecution,
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


def load_conversation_state(
    conversation: AIConversation,
) -> dict:
    """Return a validated state checkpoint or fail safely to empty state."""

    if not isinstance(conversation.state_data, dict):
        return {}

    return dict(conversation.state_data)


def save_conversation_state(
    database: Session,
    conversation: AIConversation,
    state_data: dict,
    current_intent: str | None,
    customer_id: int | None = None,
) -> AIConversation:
    """Persist the latest durable agent state for one isolated thread."""

    conversation.state_data = state_data
    conversation.current_intent = current_intent
    conversation.customer_id = customer_id
    conversation.checkpoint_version = 1
    conversation.state_updated_at = datetime.now(timezone.utc)
    database.commit()
    database.refresh(conversation)
    return conversation


def begin_request_execution(
    database: Session,
    conversation_id: int,
    request_id: str,
) -> tuple[AIRequestExecution, str | None]:
    """Claim an idempotency key or return its completed response."""

    execution = database.scalar(
        select(AIRequestExecution).where(
            AIRequestExecution.conversation_id == conversation_id,
            AIRequestExecution.request_id == request_id,
        )
    )

    if execution is not None:
        if execution.status == "completed":
            return execution, execution.response_text
        if execution.status == "executing":
            updated_at = execution.updated_at
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)

            execution_timeout = timedelta(
                minutes=get_settings().idempotency_execution_ttl_minutes
            )
            if datetime.now(timezone.utc) - updated_at <= execution_timeout:
                return execution, (
                    "This request is already being processed. Please wait "
                    "a moment before trying again."
                )

        execution.status = "executing"
        execution.error_text = None
        execution.updated_at = datetime.now(timezone.utc)
        database.commit()
        database.refresh(execution)
        return execution, None

    execution = AIRequestExecution(
        conversation_id=conversation_id,
        request_id=request_id,
        status="executing",
    )
    database.add(execution)

    try:
        database.commit()
    except IntegrityError:
        database.rollback()
        existing = database.scalar(
            select(AIRequestExecution).where(
                AIRequestExecution.conversation_id == conversation_id,
                AIRequestExecution.request_id == request_id,
            )
        )
        if existing is None:
            raise
        return existing, existing.response_text or (
            "This request is already being processed. Please wait a moment."
        )

    database.refresh(execution)
    return execution, None


def complete_request_execution(
    database: Session,
    execution: AIRequestExecution,
    response_text: str,
) -> None:
    """Store the durable result returned by an idempotent request."""

    execution.status = "completed"
    execution.response_text = response_text
    execution.error_text = None
    database.commit()


def fail_request_execution(
    database: Session,
    execution: AIRequestExecution,
    error: Exception,
) -> None:
    """Release a failed idempotency key for a safe later retry."""

    execution.status = "failed"
    execution.error_text = type(error).__name__
    database.commit()


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
