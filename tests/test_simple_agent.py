from langchain_core.messages import (
    AIMessage as LangChainAIMessage,
    HumanMessage,
)
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.ai import agent
from app.database import Base
from app.models import (
    AIConversation,
    AIEvent,
    AIMessage as DatabaseAIMessage,
)


class FakeChatModel:
    """Fake model used without making a real API request."""

    def invoke(
        self,
        messages: list[object],
    ) -> LangChainAIMessage:
        assert len(messages) >= 2

        return LangChainAIMessage(
            content=(
                "What type of appointment would you like to book?"
            )
        )


def test_simple_appointment_agent(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        agent,
        "get_chat_model",
        lambda: FakeChatModel(),
    )

    response = agent.run_appointment_agent(
        "I need an appointment."
    )

    assert response == (
        "What type of appointment would you like to book?"
    )


def test_extracts_text_from_structured_model_content(
    monkeypatch,
) -> None:
    class StructuredFakeModel:
        def invoke(
            self,
            messages: list[object],
        ) -> LangChainAIMessage:
            return LangChainAIMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Please provide your preferred service."
                        ),
                    }
                ]
            )

    monkeypatch.setattr(
        agent,
        "get_chat_model",
        lambda: StructuredFakeModel(),
    )

    response = agent.run_appointment_agent(
        "I want an appointment."
    )

    assert response == (
        "Please provide your preferred service."
    )


def test_agent_persists_messages_and_events(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
    )

    Base.metadata.create_all(bind=engine)

    test_session_local = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    monkeypatch.setattr(
        agent,
        "SessionLocal",
        test_session_local,
    )

    monkeypatch.setattr(
        agent,
        "get_chat_model",
        lambda: FakeChatModel(),
    )

    response = agent.run_appointment_agent(
        user_message="I need an appointment.",
        thread_id="thread-test-agent-001",
        request_id="request-test-agent-001",
    )

    assert response == (
        "What type of appointment would you like to book?"
    )

    with Session(engine) as database:
        conversation = database.scalar(
            select(AIConversation).where(
                AIConversation.thread_id
                == "thread-test-agent-001"
            )
        )

        assert conversation is not None

        messages = list(
            database.scalars(
                select(DatabaseAIMessage)
                .where(
                    DatabaseAIMessage.conversation_id
                    == conversation.id
                )
                .order_by(
                    DatabaseAIMessage.id
                )
            )
        )

        events = list(
            database.scalars(
                select(AIEvent)
                .where(
                    AIEvent.conversation_id
                    == conversation.id
                )
                .order_by(
                    AIEvent.id
                )
            )
        )

        assert len(messages) == 2

        assert messages[0].content == (
            "I need an appointment."
        )

        assert messages[1].content == (
            "What type of appointment would you like to book?"
        )

        assert len(events) == 2

        assert events[0].event_name == (
            "appointment_agent_model_request"
        )

        assert events[1].event_name == (
            "appointment_agent_model_response"
        )

        assert events[0].request_id == (
            "request-test-agent-001"
        )

        assert events[1].request_id == (
            "request-test-agent-001"
        )

def test_same_thread_preserves_conversation_history(
    monkeypatch,
) -> None:
    class MemoryAwareFakeModel:
        def invoke(
            self,
            messages: list[object],
        ) -> LangChainAIMessage:
            human_messages = [
                message.content
                for message in messages
                if isinstance(message, HumanMessage)
            ]

            latest_message = human_messages[-1]

            if latest_message == "My name is Kivi.":
                return LangChainAIMessage(
                    content="I will remember that your name is Kivi."
                )

            if latest_message == "What is my name?":
                assert "My name is Kivi." in human_messages

                return LangChainAIMessage(
                    content="Your name is Kivi."
                )

            return LangChainAIMessage(
                content="I do not know."
            )

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
    )

    Base.metadata.create_all(bind=engine)

    test_session_local = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    monkeypatch.setattr(
        agent,
        "SessionLocal",
        test_session_local,
    )

    monkeypatch.setattr(
        agent,
        "get_chat_model",
        lambda: MemoryAwareFakeModel(),
    )

    thread_id = "thread-memory-test-001"

    first_response = agent.run_appointment_agent(
        user_message="My name is Kivi.",
        thread_id=thread_id,
    )

    second_response = agent.run_appointment_agent(
        user_message="What is my name?",
        thread_id=thread_id,
    )

    assert first_response == (
        "I will remember that your name is Kivi."
    )

    assert second_response == "Your name is Kivi."

    with Session(engine) as database:
        conversation = database.scalar(
            select(AIConversation).where(
                AIConversation.thread_id == thread_id
            )
        )

        assert conversation is not None

        messages = list(
            database.scalars(
                select(DatabaseAIMessage)
                .where(
                    DatabaseAIMessage.conversation_id
                    == conversation.id
                )
                .order_by(DatabaseAIMessage.id)
            )
        )

        events = list(
            database.scalars(
                select(AIEvent)
                .where(
                    AIEvent.conversation_id
                    == conversation.id
                )
                .order_by(AIEvent.id)
            )
        )

        assert len(messages) == 4
        assert len(events) == 4

        assert [message.content for message in messages] == [
            "My name is Kivi.",
            "I will remember that your name is Kivi.",
            "What is my name?",
            "Your name is Kivi.",
        ]