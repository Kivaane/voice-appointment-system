import re
from typing import Annotated, Literal, TypedDict
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
from langgraph.prebuilt import ToolNode, tools_condition

from app.ai.model import get_chat_model
from app.ai.tools import (
    check_available_slots,
    list_available_services,
)
from app.ai_persistence import (
    get_or_create_conversation,
    record_event,
    save_message,
)
from app.config import get_settings
from app.database import SessionLocal
from app.models import AIEventType, AIMessageRole


READ_ONLY_TOOLS = [
    list_available_services,
    check_available_slots,
]


SYSTEM_PROMPT = """
You are an appointment assistant.

Your responsibilities are to:
- answer questions about appointment services
- use the service tool when the customer asks what services are offered
- use the availability tool when the customer asks about available slots
- ask for missing information clearly
- remember relevant details from earlier messages in the same thread
- remain concise and professional
- never invent appointment availability
- never claim that an appointment is booked unless a booking tool confirms it

The availability tool requires:
- service_id
- requested_date in YYYY-MM-DD format
- optional staff_id

Booking, cancellation, and rescheduling actions are not available yet.
""".strip()


AppointmentIntent = Literal[
    "book_appointment",
    "cancel_appointment",
    "reschedule_appointment",
    "check_availability",
    "list_services",
    "general_question",
]


ConfirmationStatus = Literal[
    "not_requested",
    "pending",
    "confirmed",
    "rejected",
]


class AppointmentAgentState(TypedDict, total=False):
    """Structured conversation state stored by LangGraph."""

    messages: Annotated[list[AnyMessage], add_messages]

    intent: AppointmentIntent | None

    customer_id: int | None
    service_id: int | None
    staff_id: int | None
    requested_date: str | None
    slot_id: int | None

    appointment_id: int | None
    cancellation_reason: str | None

    missing_fields: list[str]
    confirmation_status: ConfirmationStatus


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


def get_latest_user_message(
    state: AppointmentAgentState,
) -> str:
    """Return the latest human message from the conversation."""

    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return extract_text_content(message.content)

    return ""


def detect_intent(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Identify the user's current appointment intent."""

    user_message = get_latest_user_message(state).lower()

    if any(
        keyword in user_message
        for keyword in (
            "reschedule",
            "move my appointment",
            "change my appointment",
            "change the appointment",
        )
    ):
        intent: AppointmentIntent = "reschedule_appointment"

    elif any(
        keyword in user_message
        for keyword in (
            "cancel",
            "remove my appointment",
        )
    ):
        intent = "cancel_appointment"

    elif any(
        keyword in user_message
        for keyword in (
            "available slot",
            "available slots",
            "available time",
            "availability",
            "free slot",
            "what time",
            "which time",
        )
    ):
        intent = "check_availability"

    elif any(
        keyword in user_message
        for keyword in (
            "what services",
            "which services",
            "services available",
            "services offered",
            "what do you offer",
        )
    ):
        intent = "list_services"

    elif any(
        keyword in user_message
        for keyword in (
            "book",
            "make an appointment",
            "need an appointment",
            "schedule an appointment",
        )
    ):
        intent = "book_appointment"

    else:
        existing_intent = state.get("intent")

        if existing_intent in {
            "book_appointment",
            "cancel_appointment",
            "reschedule_appointment",
            "check_availability",
        }:
            intent = existing_intent
        else:
            intent = "general_question"

    return {
        "intent": intent,
    }


def extract_details(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Extract clearly stated appointment details from the latest message."""

    user_message = get_latest_user_message(state)
    extracted: AppointmentAgentState = {}

    id_patterns = {
        "customer_id": (
            r"\bcustomer(?:\s+id)?\s*(?:is|=|:)?\s*(\d+)\b"
        ),
        "service_id": (
            r"\bservice(?:\s+id)?\s*(?:is|=|:)?\s*(\d+)\b"
        ),
        "staff_id": (
            r"\bstaff(?:\s+id)?\s*(?:is|=|:)?\s*(\d+)\b"
        ),
        "slot_id": (
            r"\bslot(?:\s+id)?\s*(?:is|=|:)?\s*(\d+)\b"
        ),
        "appointment_id": (
            r"\bappointment(?:\s+id)?\s*(?:is|=|:)?\s*(\d+)\b"
        ),
    }

    for field_name, pattern in id_patterns.items():
        match = re.search(
            pattern,
            user_message,
            flags=re.IGNORECASE,
        )

        if match is not None:
            extracted[field_name] = int(match.group(1))

    date_match = re.search(
        r"\b\d{4}-\d{2}-\d{2}\b",
        user_message,
    )

    if date_match is not None:
        extracted["requested_date"] = date_match.group(0)

    if state.get("intent") == "cancel_appointment":
        reason_match = re.search(
            r"\b(?:because|reason(?:\s+is)?[:]?)\s+(.+)$",
            user_message,
            flags=re.IGNORECASE,
        )

        if reason_match is not None:
            extracted["cancellation_reason"] = (
                reason_match.group(1).strip().rstrip(".")
            )

    return extracted


def calculate_missing_fields(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Calculate the information still required for the current intent."""

    intent = state.get("intent")
    missing_fields: list[str] = []

    if intent == "book_appointment":
        required_fields = (
            "customer_id",
            "service_id",
            "staff_id",
            "slot_id",
        )

    elif intent == "check_availability":
        required_fields = (
            "service_id",
            "requested_date",
        )

    elif intent == "cancel_appointment":
        required_fields = (
            "appointment_id",
            "cancellation_reason",
        )

    elif intent == "reschedule_appointment":
        required_fields = (
            "appointment_id",
            "slot_id",
        )

    else:
        required_fields = ()

    for field_name in required_fields:
        if state.get(field_name) is None:
            missing_fields.append(field_name)

    return {
        "missing_fields": missing_fields,
    }


def call_model(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Generate a response and allow read-only tool calls."""

    model = get_chat_model()

    if hasattr(model, "bind_tools"):
        model = model.bind_tools(READ_ONLY_TOOLS)

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
    """Build the stateful appointment agent with tools."""

    graph_builder = StateGraph(AppointmentAgentState)

    graph_builder.add_node(
        "detect_intent",
        detect_intent,
    )

    graph_builder.add_node(
        "extract_details",
        extract_details,
    )

    graph_builder.add_node(
        "calculate_missing_fields",
        calculate_missing_fields,
    )

    graph_builder.add_node(
        "call_model",
        call_model,
    )

    graph_builder.add_node(
        "tools",
        ToolNode(READ_ONLY_TOOLS),
    )

    graph_builder.add_edge(
        START,
        "detect_intent",
    )

    graph_builder.add_edge(
        "detect_intent",
        "extract_details",
    )

    graph_builder.add_edge(
        "extract_details",
        "calculate_missing_fields",
    )

    graph_builder.add_edge(
        "calculate_missing_fields",
        "call_model",
    )

    graph_builder.add_conditional_edges(
        "call_model",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    graph_builder.add_edge(
        "tools",
        "call_model",
    )

    return graph_builder.compile(
        checkpointer=InMemorySaver(),
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