from typing import TypedDict
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.ai.model import get_chat_model
from app.ai_persistence import (
    get_or_create_conversation,
    record_event,
    save_message,
)
from app.config import get_settings
from app.database import SessionLocal
from app.models import AIEventType, AIMessageRole


SYSTEM_PROMPT = """
You are an appointment assistant.

Your responsibilities are to:
- answer simple questions about appointment booking
- ask for missing information clearly
- remain concise and professional
- never invent appointment availability
- never claim that an appointment is booked unless a booking tool confirms it

At this stage, you do not have access to booking tools.
""".strip()


class AppointmentAgentState(TypedDict):
    """State passed between nodes in the appointment graph."""

    user_message: str
    assistant_response: str


def extract_text_content(content: object) -> str:
    """Convert model response content into plain text."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")

                if isinstance(text, str):
                    text_parts.append(text)

        if text_parts:
            return "\n".join(text_parts)

    return str(content)


def call_model(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Send the user message to the configured chat model."""

    model = get_chat_model()

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["user_message"]),
        ]
    )

    return {
        "user_message": state["user_message"],
        "assistant_response": extract_text_content(response.content),
    }


def build_appointment_agent():
    """Build and compile the first appointment LangGraph."""

    graph_builder = StateGraph(AppointmentAgentState)

    graph_builder.add_node(
        "call_model",
        call_model,
    )

    graph_builder.add_edge(
        START,
        "call_model",
    )

    graph_builder.add_edge(
        "call_model",
        END,
    )

    return graph_builder.compile()


appointment_agent = build_appointment_agent()


def run_appointment_agent(
    user_message: str,
    thread_id: str | None = None,
    request_id: str | None = None,
) -> str:
    """Run one message through the graph and persist the interaction."""

    resolved_thread_id = thread_id or str(uuid4())
    database = SessionLocal()

    try:
        conversation = get_or_create_conversation(
            database=database,
            thread_id=resolved_thread_id,
        )

        save_message(
            database=database,
            conversation_id=conversation.id,
            role=AIMessageRole.USER,
            content=user_message,
        )

        record_event(
            database=database,
            conversation_id=conversation.id,
            event_type=AIEventType.MODEL_REQUEST,
            event_name="appointment_agent_model_request",
            event_data={
                "message_length": len(user_message),
            },
            request_id=request_id,
        )

        result = appointment_agent.invoke(
            {
                "user_message": user_message,
                "assistant_response": "",
            }
        )

        assistant_response = result["assistant_response"]
        settings = get_settings()

        save_message(
            database=database,
            conversation_id=conversation.id,
            role=AIMessageRole.ASSISTANT,
            content=assistant_response,
            model_name=(
                settings.gemini_model
                if settings.ai_provider == "gemini"
                else settings.openai_model
            ),
        )

        record_event(
            database=database,
            conversation_id=conversation.id,
            event_type=AIEventType.MODEL_RESPONSE,
            event_name="appointment_agent_model_response",
            event_data={
                "response_length": len(assistant_response),
            },
            request_id=request_id,
        )

        return assistant_response

    finally:
        database.close()