from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.ai import agent
from app.ai_persistence import (
    get_or_create_conversation,
    load_conversation_state,
    save_conversation_state,
)
from app.database import Base


def build_test_sessions():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def test_state_checkpoint_round_trip_and_thread_isolation() -> None:
    engine, session_local = build_test_sessions()

    with Session(engine) as database:
        first = get_or_create_conversation(database, "thread-one")
        second = get_or_create_conversation(database, "thread-two")
        save_conversation_state(
            database,
            first,
            {
                "intent": "book_appointment",
                "service_id": 2,
                "requested_date": "2026-08-06",
            },
            "book_appointment",
        )

        assert load_conversation_state(first)["service_id"] == 2
        assert load_conversation_state(second) == {}

    session_local().close()


def test_corrupt_checkpoint_fails_safely() -> None:
    engine, _ = build_test_sessions()

    with Session(engine) as database:
        conversation = get_or_create_conversation(database, "corrupt-thread")
        conversation.state_data = ["not", "a", "mapping"]

        assert load_conversation_state(conversation) == {}


def test_recreated_agent_continues_same_thread(monkeypatch) -> None:
    _, session_local = build_test_sessions()
    monkeypatch.setattr(agent, "SessionLocal", session_local)

    class HistoryModel:
        def invoke(self, messages):
            human_messages = [
                message.content
                for message in messages
                if isinstance(message, HumanMessage)
            ]
            if human_messages[-1] == "My code word is mango.":
                return AIMessage(content="I will remember mango.")
            assert "My code word is mango." in human_messages
            return AIMessage(content="Your code word is mango.")

    monkeypatch.setattr(agent, "get_chat_model", lambda: HistoryModel())
    thread_id = "durable-restart-thread"

    first = agent.run_appointment_agent(
        "My code word is mango.",
        thread_id=thread_id,
    )
    monkeypatch.setattr(
        agent,
        "persistent_appointment_agent",
        agent.build_appointment_agent(),
    )
    second = agent.run_appointment_agent(
        "What is my code word?",
        thread_id=thread_id,
    )

    assert first == "I will remember mango."
    assert second == "Your code word is mango."
