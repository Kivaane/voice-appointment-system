"""Shared LangGraph state types and appointment-agent constants."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


AppointmentIntent = Literal[
    "book_appointment",
    "cancel_appointment",
    "reschedule_appointment",
    "check_availability",
    "view_appointments",
    "list_services",
    "general_question",
]


ConfirmationStatus = Literal[
    "not_requested",
    "pending",
    "confirmed",
    "rejected",
]


class TimePreference(TypedDict, total=False):
    """A deterministic time constraint extracted from user text."""

    kind: Literal[
        "period",
        "after",
        "before",
        "exact",
        "earliest",
        "latest",
    ]
    minute_of_day: int
    period: Literal["morning", "afternoon", "evening"]
    label: str


ACTIVE_FLOW_INTENTS = {
    "book_appointment",
    "reschedule_appointment",
    "cancel_appointment",
    "check_availability",
}


INFORMATION_INTERRUPTION_INTENTS = {
    "ask_notification_capability",
    "ask_service_availability",
    "ask_pricing",
    "ask_duration",
    "ask_service_list",
    "ask_opening_hours",
    "ask_location",
    "ask_insurance",
    "ask_cancellation_policy",
    "ask_payment_methods",
}


class AppointmentAgentState(TypedDict, total=False):
    """Structured conversation state stored by LangGraph."""

    messages: Annotated[list[AnyMessage], add_messages]

    intent: AppointmentIntent | None

    customer_id: int | None
    customer_name: str | None
    customer_phone_number: str | None
    customer_phone_invalid: bool

    service_id: int | None
    service_name: str | None

    staff_id: int | None
    staff_name: str | None

    requested_date: str | None
    time_preference: TimePreference | None
    time_preference_error: str | None
    tool_error: str | None

    available_services: list[dict[str, object]] | None
    available_slots: list[dict[str, object]] | None
    upcoming_alternative_slots: list[dict[str, object]] | None

    selected_slot_summary: str | None
    booking_summary: str | None
    slot_id: int | None

    appointment_id: int | None
    appointment_reference_number: str | None
    appointment_status: str | None
    current_slot_id: int | None
    cancellation_reason: str | None

    slot_selection_error: str | None

    missing_fields: list[str]
    next_question: str | None
    confirmation_status: ConfirmationStatus

    transaction_updated_at: str | None
    pending_action_started_at: str | None
    slot_options_updated_at: str | None
    state_expired_message: str | None

    semantic_nlu: dict[str, object] | None
    secondary_intents: list[str]
    mixed_intent_clarification: str | None
    date_clarification: str | None
    clarification_attempts: int

    transaction_started_explicitly: bool
    paused_intent: AppointmentIntent | None