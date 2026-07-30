from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import (
    AIMessage as LangChainAIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

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
- remember relevant details from earlier messages in the same thread
- remain concise and professional
- never invent appointment availability
- never claim that an appointment is booked unless a booking tool confirms it

At this stage, you do not have access to booking tools.
""".strip()


class AppointmentAgentState(TypedDict):
    """Conversation state stored by LangGraph."""

    messages: Annotated[list[AnyMessage], add_messages]


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
    """Generate a response using the complete thread history."""

    model = get_chat_model()

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]
    )

    return {
        "messages": [response],
    }


def build_appointment_agent():
    """Build the stateful appointment LangGraph."""

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

    checkpointer = InMemorySaver()

    return graph_builder.compile(
        checkpointer=checkpointer,
    )


appointment_agent = build_appointment_agent()


def run_appointment_agent(
    user_message: str,
    thread_id: str | None = None,
    request_id: str | None = None,
) -> str:
    """Run and persist one turn of a stateful conversation."""

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
                "messages": [
                    HumanMessage(content=user_message),
                ]
            },
            config={
                "configurable": {
                    "thread_id": resolved_thread_id,
                }
            },
        )

        final_message = result["messages"][-1]

        if not isinstance(final_message, LangChainAIMessage):
            raise RuntimeError(
                "The appointment agent did not return an AI message."
            )

        assistant_response = extract_text_content(
            final_message.content
        )

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