import re
from datetime import date, datetime, timedelta, timezone
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

from app.ai.model import classify_unknown_message, get_chat_model
from app.ai.prompts import APPOINTMENT_SYSTEM_PROMPT
from app.appointment_services import (
    AppointmentConflictError,
    AppointmentNotFoundError,
    InvalidAppointmentError,
    cancel_appointment,
    create_appointment,
    reschedule_appointment,
)
from app.ai.tools import (
    check_available_slots,
    list_available_services,
)
from app.ai_persistence import (
    begin_request_execution,
    complete_request_execution,
    fail_request_execution,
    get_or_create_conversation,
    list_conversation_messages,
    load_conversation_state,
    record_event,
    save_conversation_state,
    save_message,
)
from app.config import get_settings
from app.database import SessionLocal
from app.models import (
    AIEventType,
    AIMessageRole,
    Appointment,
    AppointmentStatus,
    Customer,
    Service,
    Staff,
)
from app.schemas import AppointmentCreate
from app.time_utils import (
    business_now_naive,
    business_today,
    to_business_datetime,
    utc_now,
)
from app.ai.nlu import NLUResult, classify_message, normalize_domain_typos

READ_ONLY_TOOLS = [
    list_available_services,
    check_available_slots,
]


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

    kind: Literal["period", "after", "before", "exact", "earliest", "latest"]
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


def informational_response_intent(
    state: AppointmentAgentState,
) -> AppointmentIntent:
    """Preserve an active flow while answering a temporary question."""

    current_intent = state.get("intent")

    if (
        current_intent in ACTIVE_FLOW_INTENTS
        and state.get("transaction_started_explicitly") is not False
    ):
        return current_intent

    return "general_question"


def semantic_fallback_is_safe(
    state: AppointmentAgentState,
    user_message: str,
) -> bool:
    """Allow semantic classification only for unknown, low-risk language."""

    if (
        is_human_handoff_message(user_message)
        or is_graceful_exit_message(user_message)
        or is_capabilities_request(user_message)
        or is_greeting_or_small_talk(user_message)
        or is_upcoming_availability_request(user_message)
        or is_appointment_listing_request(user_message)
    ):
        return False
    if state.get("confirmation_status") == "pending":
        return False
    if is_confirmation_yes_message(user_message) or is_confirmation_no_message(
        user_message
    ):
        return False
    if extract_appointment_reference(user_message) is not None:
        return False
    if extract_phone_candidate(user_message) is not None:
        return False
    if find_slot_from_message(
        user_message,
        state.get("available_slots"),
    ) is not None:
        return False
    if re.fullmatch(r"\s*\d+\s*", user_message):
        return False

    return True


def classify_message_for_state(
    state: AppointmentAgentState,
    user_message: str,
) -> tuple[NLUResult, dict[str, object] | None]:
    """Use deterministic NLU first and semantic NLU only when safe."""

    deterministic_result = classify_message(user_message)
    if deterministic_result.intent != "unknown":
        return deterministic_result, None
    if not semantic_fallback_is_safe(state, user_message):
        return deterministic_result, None

    try:
        model_result = classify_unknown_message(user_message)
    except Exception:
        return deterministic_result, None

    if model_result is None or model_result.confidence < 0.65:
        return deterministic_result, None

    semantic_payload: dict[str, object] = {
        **model_result.model_dump(),
        "source_message": user_message,
    }
    return NLUResult(
        intent=model_result.intent,
        confidence=model_result.confidence,
        service_hint=model_result.service_hint,
        staff_hint=model_result.staff_hint,
        date_hint=model_result.date_hint,
        time_hint=model_result.time_hint,
        customer_name=model_result.customer_name,
        phone_hint=model_result.phone_hint,
        appointment_reference=model_result.appointment_reference,
        should_start_booking=model_result.intent == "book_appointment",
        requires_clarification=model_result.requires_clarification,
        clarification_reason=model_result.clarification_reason,
    ), semantic_payload


def nlu_result_for_response(
    state: AppointmentAgentState,
    user_message: str,
) -> NLUResult:
    """Reuse the validated semantic result later in the same graph turn."""

    semantic_nlu = state.get("semantic_nlu")
    if (
        isinstance(semantic_nlu, dict)
        and semantic_nlu.get("source_message") == user_message
    ):
        return NLUResult(
            intent=str(semantic_nlu.get("intent") or "unknown"),
            confidence=float(semantic_nlu.get("confidence") or 0),
            service_hint=semantic_nlu.get("service_hint"),
            staff_hint=semantic_nlu.get("staff_hint"),
            date_hint=semantic_nlu.get("date_hint"),
            time_hint=semantic_nlu.get("time_hint"),
            customer_name=semantic_nlu.get("customer_name"),
            phone_hint=semantic_nlu.get("phone_hint"),
            appointment_reference=semantic_nlu.get(
                "appointment_reference"
            ),
            requires_clarification=bool(
                semantic_nlu.get("requires_clarification")
            ),
            clarification_reason=semantic_nlu.get(
                "clarification_reason"
            ),
        )

    return classify_message(user_message)


def entity_source_message(
    state: AppointmentAgentState,
    user_message: str,
) -> str:
    """Append only validated semantic hints for deterministic extraction."""

    semantic_nlu = state.get("semantic_nlu")
    if (
        not isinstance(semantic_nlu, dict)
        or semantic_nlu.get("source_message") != user_message
    ):
        return user_message

    hints: list[str] = []
    for field_name in (
        "service_hint",
        "staff_hint",
        "date_hint",
        "time_hint",
    ):
        value = semantic_nlu.get(field_name)
        if isinstance(value, str) and value.strip():
            hints.append(value.strip())

    customer_name = semantic_nlu.get("customer_name")
    if isinstance(customer_name, str) and customer_name.strip():
        hints.append(f"my name is {customer_name.strip()}")

    return " ".join([user_message, *hints]).strip()


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


def normalize_text(text: str) -> str:
    """Normalize text for simple matching."""

    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        normalize_domain_typos(text).lower(),
    ).strip()
    return re.sub(r"\b([ap])\s+m\b", r"\1m", normalized)


def parse_option_ordinal(user_message: str) -> int | None:
    """Parse a standalone safe service/slot option ordinal."""

    normalized = normalize_text(user_message)
    ordinal_choices = {
        "first": 1,
        "1st": 1,
        "one": 1,
        "second": 2,
        "2nd": 2,
        "two": 2,
        "third": 3,
        "3rd": 3,
        "three": 3,
        "fourth": 4,
        "4th": 4,
        "four": 4,
    }
    if normalized in ordinal_choices:
        return ordinal_choices[normalized]
    number_match = re.fullmatch(r"(\d+)", normalized)
    return int(number_match.group(1)) if number_match is not None else None


def extract_appointment_reference(
    user_message: str,
) -> str | None:
    """Extract an appointment reference such as APT-871E6728."""

    reference_match = re.search(
        r"\bAPT-[A-Z0-9]+\b",
        user_message,
        flags=re.IGNORECASE,
    )

    if reference_match is None:
        return None

    return reference_match.group(0).upper()


def get_appointment_by_reference(
    reference_number: str,
) -> dict[str, object] | None:
    """Load appointment details by public reference number."""

    database = SessionLocal()

    try:
        appointment = (
            database.query(Appointment)
            .filter(
                Appointment.reference_number == reference_number,
            )
            .first()
        )

        if appointment is None:
            return None

        return appointment_to_conversation_details(appointment)

    finally:
        database.close()


def appointment_to_conversation_details(
    appointment: Appointment,
) -> dict[str, object]:
    """Convert an appointment model into safe display/state values."""

    status = appointment.status

    return {
        "appointment_id": appointment.id,
        "appointment_reference_number": appointment.reference_number,
        "appointment_status": (
            status.value
            if hasattr(status, "value")
            else str(status)
        ),
        "customer_id": appointment.customer_id,
        "customer_phone_number": (
            appointment.customer.phone_number
            if appointment.customer is not None
            else None
        ),
        "service_id": appointment.service_id,
        "service_name": (
            appointment.service.name
            if appointment.service is not None
            else "Appointment"
        ),
        "staff_id": appointment.staff_id,
        "staff_name": (
            appointment.staff.full_name
            if appointment.staff is not None
            else "Selected staff"
        ),
        "current_slot_id": appointment.slot_id,
        "start_datetime": appointment.start_datetime.isoformat(),
        "end_datetime": appointment.end_datetime.isoformat(),
    }


def get_appointment_by_id_for_conversation(
    appointment_id: int,
) -> dict[str, object] | None:
    """Load appointment display details by internal ID."""

    database = SessionLocal()

    try:
        appointment = database.get(Appointment, appointment_id)

        if appointment is None:
            return None

        return appointment_to_conversation_details(appointment)

    finally:
        database.close()


def get_upcoming_appointments_for_customer(
    customer_id: int | None = None,
    phone_number: str | None = None,
) -> list[dict[str, object]]:
    """Load confirmed future appointments for a known customer."""

    database = SessionLocal()

    try:
        resolved_customer_id = customer_id

        if resolved_customer_id is None and phone_number is not None:
            customer = (
                database.query(Customer)
                .filter(Customer.phone_number == phone_number)
                .first()
            )

            if customer is None:
                return []

            resolved_customer_id = int(customer.id)

        if resolved_customer_id is None:
            return []

        appointments = (
            database.query(Appointment)
            .filter(
                Appointment.customer_id == resolved_customer_id,
                Appointment.start_datetime >= business_now_naive(),
                Appointment.status == AppointmentStatus.CONFIRMED,
            )
            .order_by(Appointment.start_datetime)
            .all()
        )

        return [
            appointment_to_conversation_details(appointment)
            for appointment in appointments
        ]

    finally:
        database.close()


def get_active_services() -> list[dict[str, object]]:
    """Load active services from the database for conversation use."""

    database = SessionLocal()

    try:
        services = (
            database.query(Service)
            .filter(Service.is_active.is_(True))
            .order_by(Service.id)
            .all()
        )

        return [
            {
                "id": service.id,
                "name": service.name,
                "description": service.description,
                "duration_minutes": service.duration_minutes,
                "price": (
                    float(service.price)
                    if service.price is not None
                    else None
                ),
            }
            for service in services
        ]

    finally:
        database.close()


def get_service_by_id(
    service_id: int,
) -> dict[str, object] | None:
    """Load one service by ID."""

    database = SessionLocal()

    try:
        service = (
            database.query(Service)
            .filter(Service.id == service_id)
            .first()
        )

        if service is None:
            return None

        return {
            "id": service.id,
            "name": service.name,
            "description": service.description,
            "duration_minutes": service.duration_minutes,
            "price": (
                float(service.price)
                if service.price is not None
                else None
            ),
        }

    finally:
        database.close()


def get_active_staff_for_service(
    service_id: int,
) -> list[dict[str, object]]:
    """Load active staff members who can provide a service."""

    database = SessionLocal()

    try:
        staff_members = (
            database.query(Staff)
            .join(Staff.services)
            .filter(
                Service.id == service_id,
                Staff.is_active.is_(True),
            )
            .order_by(Staff.id)
            .all()
        )

        return [
            {
                "id": staff.id,
                "full_name": staff.full_name,
                "speciality": staff.speciality,
            }
            for staff in staff_members
        ]

    finally:
        database.close()


def format_price(price: object) -> str:
    """Format a service price for display."""

    if price is None:
        return ""

    return f", LKR {float(price):,.0f}"


def format_service_options(
    services: list[dict[str, object]],
) -> str:
    """Format services as a friendly numbered list."""

    if not services:
        return "No active services are available right now."

    lines: list[str] = []

    for index, service in enumerate(services, start=1):
        description = service.get("description") or "Appointment service"
        duration = service.get("duration_minutes")
        price = format_price(service.get("price"))

        lines.append(
            f"{index}. {service['name']} — {description} "
            f"({duration} minutes{price})"
        )

    return "\n".join(lines)


def find_service_from_message(
    user_message: str,
    services: list[dict[str, object]],
    allow_numeric_choice: bool,
    prefer_last_match: bool = False,
) -> dict[str, object] | None:
    """Resolve a service from natural text or numeric choice."""

    normalized_message = normalize_text(user_message)

    if allow_numeric_choice:
        number = parse_option_ordinal(user_message)

        if number is not None:

            if 1 <= number <= len(services):
                return services[number - 1]

            for service in services:
                if service["id"] == number:
                    return service

    matched_services: list[tuple[int, dict[str, object]]] = []

    for service in services:
        service_name = str(service["name"])
        normalized_name = normalize_text(service_name)

        if normalized_name in normalized_message:
            matched_services.append(
                (normalized_message.rfind(normalized_name), service)
            )
            continue

        service_words = [
            word
            for word in normalized_name.split()
            if len(word) >= 4
        ]

        positions = [
            normalized_message.rfind(word)
            for word in service_words
            if word in normalized_message
        ]
        if positions:
            matched_services.append((max(positions), service))

        aliases = {
            "dental care": ("dental", "dentist", "tooth", "teeth"),
            "physiotherapy": ("physio", "physiotherapy"),
            "dermatology": ("dermatology", "skin"),
            "general consultation": ("general", "consultation"),
        }
        if any(
            re.search(rf"\b{re.escape(alias)}\b", normalized_message)
            for alias in aliases.get(normalized_name, ())
        ):
            matched_services.append(
                (max(normalized_message.rfind(alias) for alias in aliases[normalized_name] if alias in normalized_message), service)
            )

    if matched_services:
        if prefer_last_match:
            return max(matched_services, key=lambda match: match[0])[1]
        return matched_services[0][1]

    return None


def find_staff_from_message(
    user_message: str,
    staff_members: list[dict[str, object]],
) -> dict[str, object] | None:
    """Resolve a staff member from natural text."""

    normalized_message = normalize_text(user_message)

    for staff in staff_members:
        staff_name = str(staff["full_name"])
        normalized_name = normalize_text(staff_name)

        if normalized_name in normalized_message:
            return staff

        name_words = [
            word
            for word in normalized_name.split()
            if len(word) >= 4
        ]

        if any(word in normalized_message for word in name_words):
            return staff

    return None


def parse_requested_date(
    user_message: str,
) -> str | None:
    """Parse simple natural dates into YYYY-MM-DD."""

    iso_date_match = re.search(
        r"\b\d{4}-\d{2}-\d{2}\b",
        user_message,
    )

    if iso_date_match is not None:
        return iso_date_match.group(0)

    normalized_message = normalize_text(user_message)
    today = business_today()

    month_names = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    named_date_match = re.search(
        r"\b(0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\s+"
        r"(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)"
        r"(?:\s+(\d{4}))?\b",
        normalized_message,
    )

    if named_date_match is not None:
        requested_day = int(named_date_match.group(1))
        requested_month = month_names[named_date_match.group(2)]
        requested_year = int(named_date_match.group(3) or today.year)

        try:
            requested_date = date(
                requested_year,
                requested_month,
                requested_day,
            )
        except ValueError:
            return None

        if named_date_match.group(3) is None and requested_date < today:
            requested_date = requested_date.replace(year=today.year + 1)

        return requested_date.isoformat()

    if "day after tomorrow" in normalized_message:
        return (today + timedelta(days=2)).isoformat()

    if "tomorrow" in normalized_message:
        return (today + timedelta(days=1)).isoformat()

    if "today" in normalized_message:
        return today.isoformat()

    ordinal_day_match = re.search(
        r"\b(0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)\b",
        user_message,
        flags=re.IGNORECASE,
    )

    contextual_day_match = re.search(
        r"\b(?:on|for|date|day)\s+"
        r"(0?[1-9]|[12][0-9]|3[01])\b",
        user_message,
        flags=re.IGNORECASE,
    )

    day_number_match = ordinal_day_match or contextual_day_match

    if day_number_match is not None:
        requested_day = int(day_number_match.group(1))

        try:
            requested_date = today.replace(day=requested_day)
        except ValueError:
            requested_date = None

        if requested_date is not None:
            if requested_date < today:
                if today.month == 12:
                    requested_date = requested_date.replace(
                        year=today.year + 1,
                        month=1,
                    )
                else:
                    requested_date = requested_date.replace(
                        month=today.month + 1,
                    )

            return requested_date.isoformat()

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    for weekday_name, weekday_number in weekdays.items():
        if weekday_name in normalized_message:
            days_ahead = (
                weekday_number - today.weekday()
            ) % 7

            if days_ahead == 0:
                days_ahead += 7

            return (today + timedelta(days=days_ahead)).isoformat()

    return None


def parse_clock_time(
    hour_text: str,
    minute_text: str | None,
    meridiem: str | None,
) -> int | None:
    """Convert a user clock value into minutes after midnight."""

    hour = int(hour_text)
    minute = int(minute_text or 0)

    if minute > 59:
        return None

    if meridiem is not None:
        if not 1 <= hour <= 12:
            return None
        hour %= 12
        if meridiem.lower() == "pm":
            hour += 12
    elif not 0 <= hour <= 23:
        return None

    return hour * 60 + minute


def extract_time_preference(
    user_message: str,
) -> tuple[TimePreference | None, str | None]:
    """Extract a period or clock constraint without selecting a slot."""

    normalized_message = normalize_text(user_message)

    for period in ("morning", "afternoon", "evening"):
        if re.search(rf"\b{period}\b", normalized_message):
            return {
                "kind": "period",
                "period": period,
                "label": period,
            }, None

    if re.search(r"\bearliest(?:\s+available)?(?:\s+slot)?\b", normalized_message):
        return {"kind": "earliest", "label": "earliest available"}, None

    if re.search(r"\blatest(?:\s+available)?(?:\s+slot)?\b", normalized_message):
        return {"kind": "latest", "label": "latest available"}, None

    if "before noon" in normalized_message:
        return {
            "kind": "before",
            "minute_of_day": 12 * 60,
            "label": "before noon",
        }, None

    number_words = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
    }
    for word, number in number_words.items():
        normalized_message = re.sub(
            rf"\b(after|before|at)\s+{word}\b",
            rf"\1 {number}",
            normalized_message,
        )

    if re.search(r"\b(?:around|about)\s+\d{1,2}\b", normalized_message):
        if re.search(r"\b(?:am|pm)\b", normalized_message) is None:
            return None, (
                "Please clarify the time, for example 2:00 PM, morning, "
                "afternoon, before 11:00 AM, or after 2:00 PM."
            )

    clock_match = re.search(
        r"\b(after|before|at)\s+"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        normalized_message,
    )

    if clock_match is not None:
        qualifier = clock_match.group(1).lower()
        meridiem = clock_match.group(4)

        if meridiem is None and int(clock_match.group(2)) <= 12:
            return None, (
                "Please clarify whether that time is AM or PM."
            )

        minute_of_day = parse_clock_time(
            clock_match.group(2),
            clock_match.group(3),
            meridiem,
        )

        if minute_of_day is None:
            return None, "Please provide a valid appointment time."

        kind = {
            "after": "after",
            "before": "before",
            "at": "exact",
        }[qualifier]

        return {
            "kind": kind,
            "minute_of_day": minute_of_day,
            "label": clock_match.group(0),
        }, None

    return None, None


def filter_slots_by_time_preference(
    slots: list[dict[str, object]],
    preference: TimePreference | None,
) -> list[dict[str, object]]:
    """Return only slots matching a deterministic time preference."""

    if preference is None:
        return list(slots)

    kind = preference.get("kind")
    ordered_slots = sorted(
        slots,
        key=lambda slot: str(slot.get("start_datetime") or ""),
    )
    if kind == "earliest":
        return ordered_slots[:1]
    if kind == "latest":
        return ordered_slots[-1:]

    matching_slots: list[dict[str, object]] = []

    for slot in slots:
        start_datetime = parse_datetime(slot.get("start_datetime"))

        if start_datetime is None:
            continue

        minute_of_day = start_datetime.hour * 60 + start_datetime.minute
        if kind == "period":
            ranges = {
                "morning": (5 * 60, 12 * 60),
                "afternoon": (12 * 60, 17 * 60),
                "evening": (17 * 60, 22 * 60),
            }
            start_minute, end_minute = ranges[str(preference["period"])]
            matches = start_minute <= minute_of_day < end_minute
        elif kind == "after":
            matches = minute_of_day >= int(preference["minute_of_day"])
        elif kind == "before":
            matches = minute_of_day < int(preference["minute_of_day"])
        else:
            matches = minute_of_day == int(preference["minute_of_day"])

        if matches:
            matching_slots.append(slot)

    return matching_slots

def parse_datetime(value: object) -> datetime | None:
    """Parse an ISO datetime safely."""

    if not isinstance(value, str):
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def format_time(value: object) -> str:
    """Format an ISO datetime as a friendly time."""

    parsed_value = parse_datetime(value)

    if parsed_value is None:
        return str(value)

    business_value = to_business_datetime(parsed_value)
    hour = business_value.strftime("%I").lstrip("0")
    minute = business_value.strftime("%M")
    meridiem = business_value.strftime("%p")

    return f"{hour}:{minute} {meridiem}"


def format_date(value: str | None) -> str:
    """Format YYYY-MM-DD into a friendly date."""

    if value is None:
        return "that date"

    try:
        parsed_date = date.fromisoformat(value)
    except ValueError:
        return value

    return parsed_date.strftime("%A, %d %B %Y")


def format_slot_summary(
    slot: dict[str, object],
) -> str:
    """Format one selected slot."""

    staff_name = slot.get("staff_name") or f"staff {slot.get('staff_id')}"
    service_name = slot.get("service_name") or "appointment"

    return (
        f"{service_name} at {format_time(slot.get('start_datetime'))} "
        f"with {staff_name}"
    )


def format_available_slots(
    slots: list[dict[str, object]],
) -> str:
    """Format available slots as friendly numbered options."""

    lines: list[str] = []

    for index, slot in enumerate(slots, start=1):
        staff_name = (
            slot.get("staff_name")
            or f"staff {slot.get('staff_id')}"
        )

        lines.append(
            f"{index}. {format_time(slot.get('start_datetime'))} – "
            f"{format_time(slot.get('end_datetime'))} "
            f"with {staff_name}"
        )

    return "\n".join(lines)


def format_resume_prompt(state: AppointmentAgentState) -> str | None:
    """Return one concise prompt that resumes an interrupted transaction."""

    intent = state.get("intent")
    if (
        intent not in ACTIVE_FLOW_INTENTS
        or state.get("transaction_started_explicitly") is False
    ):
        return None

    if (
        state.get("confirmation_status") == "pending"
        and state.get("booking_summary") is not None
    ):
        return "Your pending request is unchanged. Reply yes to confirm or no to change it."

    slots = state.get("available_slots") or []
    if slots and state.get("slot_id") is None:
        times = [format_time(slot.get("start_datetime")) for slot in slots[:3]]
        if len(times) == 1:
            choices = times[0]
        elif len(times) == 2:
            choices = f"{times[0]} or {times[1]}"
        else:
            choices = ", ".join(times[:-1]) + f", or {times[-1]}"
        return f"Would you prefer {choices}?"

    if intent in {"book_appointment", "check_availability"}:
        if state.get("service_id") is None:
            return None
        if state.get("requested_date") is None:
            return "Which date would you prefer?"
        if intent == "book_appointment" and state.get("slot_id") is None:
            return (
                f"You were selecting a slot for "
                f"{format_date(state.get('requested_date'))}. "
                "Would you like me to continue?"
            )
        if intent == "book_appointment" and state.get("customer_name") is None:
            return "What name should I use for the booking?"
        if (
            intent == "book_appointment"
            and state.get("customer_phone_number") is None
        ):
            return "What Sri Lankan phone number should I use?"

    if intent == "cancel_appointment":
        if state.get("appointment_id") is None:
            return "Please share the appointment reference or phone number."
        return "Should I continue with the cancellation?"

    if intent == "reschedule_appointment":
        if state.get("appointment_id") is None:
            return "Please share the appointment reference or phone number."
        if state.get("requested_date") is None:
            return "Which new date would you prefer?"

    return None


def compose_informational_response(
    answer: str,
    state: AppointmentAgentState,
) -> str:
    """Answer first, then include at most one contextual resume question."""

    resume_prompt = format_resume_prompt(state)
    if resume_prompt is None:
        return answer
    return f"{answer} {resume_prompt}"


def clarification_response(
    state: AppointmentAgentState,
    message: str,
) -> AppointmentAgentState:
    """Track failed clarification turns and offer handoff after three."""

    attempts = int(state.get("clarification_attempts") or 0) + 1
    if attempts >= 3:
        message = (
            f"{message} If you prefer, please contact the front desk and "
            "a staff member can help you continue."
        )
    return {
        "next_question": message,
        "clarification_attempts": attempts,
    }


def get_date_clarification_message(user_message: str) -> str | None:
    """Return concrete choices when the user supplies competing dates."""

    normalized_message = normalize_text(user_message)
    weekdays = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )
    mentioned = [day for day in weekdays if day in normalized_message]
    relative_date: str | None = None

    if "day after tomorrow" in normalized_message:
        relative_date = (
            business_today() + timedelta(days=2)
        ).isoformat()

    elif re.search(r"\btomorrow\b", normalized_message):
        relative_date = (
            business_today() + timedelta(days=1)
        ).isoformat()

    elif re.search(r"\btoday\b", normalized_message):
        relative_date = business_today().isoformat()

    if relative_date is not None and mentioned:
        weekday_date = parse_requested_date(
            f"next {mentioned[0]}"
        )

        if (
            weekday_date is not None
            and weekday_date != relative_date
        ):
            return (
                f"Do you mean {format_date(relative_date)} "
                f"or {format_date(weekday_date)}?"
            )
    if len(mentioned) >= 2:
        choices = [
            format_date(parse_requested_date(f"next {day}"))
            for day in mentioned[:2]
        ]
        return f"Do you mean {choices[0]} or {choices[1]}?"

    if "next week" in normalized_message and not mentioned:
        monday = format_date(parse_requested_date("next monday"))
        sunday = format_date(parse_requested_date("next sunday"))
        return (
            f"Which day next week would you prefer, between {monday} "
            f"and {sunday}?"
        )

    return None


def format_upcoming_available_slots(
    service_name: str,
    slots: list[dict[str, object]],
) -> str:
    """Format upcoming slots as a friendly list grouped by date."""

    if not slots:
        return (
            f"I checked upcoming {service_name} availability, but "
            "there are no available slots in the current demo data. "
            "You can try another service or ask staff to add more "
            "availability."
        )

    sorted_slots = sorted(
        slots,
        key=lambda slot: str(slot.get("start_datetime") or ""),
    )
    lines = [
        f"I checked upcoming {service_name} availability.",
        "",
        "Next available slots:",
    ]
    current_date: str | None = None

    for index, slot in enumerate(sorted_slots, start=1):
        parsed_start = parse_datetime(
            slot.get("start_datetime"),
        )
        slot_date = (
            parsed_start.date().isoformat()
            if parsed_start is not None
            else ""
        )

        if slot_date != current_date:
            current_date = slot_date
            lines.extend(
                [
                    "",
                    f"{format_date(slot_date)}:",
                ]
            )

        start_time = format_time(slot.get("start_datetime"))
        end_time = format_time(slot.get("end_datetime"))
        staff_name = slot.get("staff_name") or "available staff"

        lines.append(
            f"{index}. {start_time} – {end_time} with {staff_name}"
        )

    return "\n".join(lines)


def format_appointment_details(
    appointment: dict[str, object],
    index: int | None = None,
) -> str:
    """Format one appointment for listing, status, or cancellation."""

    prefix = f"{index}. " if index is not None else ""
    status = str(appointment.get("appointment_status") or "UNKNOWN")
    start_datetime = appointment.get("start_datetime")
    end_datetime = appointment.get("end_datetime")
    parsed_start = parse_datetime(start_datetime)
    appointment_date = (
        parsed_start.date().isoformat()
        if parsed_start is not None
        else None
    )

    return (
        f"{prefix}Reference: "
        f"{appointment.get('appointment_reference_number')}\n"
        f"Service: {appointment.get('service_name') or 'Appointment'}\n"
        f"Doctor/Staff: "
        f"{appointment.get('staff_name') or 'Selected staff'}\n"
        f"Date: {format_date(appointment_date)}\n"
        f"Time: {format_time(start_datetime)} – "
        f"{format_time(end_datetime)}\n"
        f"Status: {status}"
    )


def format_appointment_list(
    appointments: list[dict[str, object]],
) -> str:
    """Format upcoming appointments as a controlled response."""

    if not appointments:
        return (
            "I couldn't find any upcoming appointments for those "
            "details."
        )

    formatted_appointments = [
        format_appointment_details(appointment, index)
        for index, appointment in enumerate(appointments, start=1)
    ]

    return (
        "Here are your upcoming appointments:\n\n"
        + "\n\n".join(formatted_appointments)
    )


def format_cancellation_confirmation_summary(
    appointment: dict[str, object],
) -> str:
    """Format a pending cancellation without mutating the database."""

    return (
        "Please confirm the appointment you want to cancel:\n\n"
        + format_appointment_details(appointment)
        + "\n\nShould I cancel this appointment?"
    )


def find_slot_from_message(
    user_message: str,
    available_slots: list[dict[str, object]] | None,
) -> dict[str, object] | None:
    """Resolve a chosen slot from natural text."""

    if not available_slots:
        return None

    normalized_message = normalize_text(user_message)

    slot_id_match = re.search(
        r"\bslot\s+id\s*(?:is|=|:)?\s*(\d+)\b",
        user_message,
        flags=re.IGNORECASE,
    )

    if slot_id_match is not None:
        selected_slot_id = int(slot_id_match.group(1))

        for slot in available_slots:
            if slot.get("slot_id") == selected_slot_id:
                return slot

    option_match = re.search(
        r"\b(?:option|slot)\s*(\d+)\b",
        normalized_message,
    )

    if option_match is not None:
        option_number = int(option_match.group(1))

        if 1 <= option_number <= len(available_slots):
            return available_slots[option_number - 1]

    single_slot_phrases = (
        "this slot",
        "this one",
        "that one",
        "take this",
        "choose this",
        "select this",
        "yes this slot",
        "book this slot",
    )

    if (
        len(available_slots) == 1
        and any(
            phrase in normalized_message
            for phrase in single_slot_phrases
        )
    ):
        return available_slots[0]

    ordinal_choices = {
        "first": 1,
        "1st": 1,
        "one": 1,
        "second": 2,
        "2nd": 2,
        "two": 2,
        "third": 3,
        "3rd": 3,
        "three": 3,
        "fourth": 4,
        "4th": 4,
        "four": 4,
    }

    for word, option_number in ordinal_choices.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized_message):
            if 1 <= option_number <= len(available_slots):
                return available_slots[option_number - 1]

    bare_number_match = re.fullmatch(
        r"\s*(\d+)\s*\.?\s*",
        user_message,
    )

    if bare_number_match is not None:
        number = int(bare_number_match.group(1))

        if 1 <= number <= len(available_slots):
            return available_slots[number - 1]

    for slot in available_slots:
        start_datetime = parse_datetime(
            slot.get("start_datetime"),
        )

        if start_datetime is None:
            continue

        hour_24 = str(start_datetime.hour)
        hour_12 = str(
            start_datetime.hour
            if start_datetime.hour <= 12
            else start_datetime.hour - 12
        )
        minute = f"{start_datetime.minute:02d}"

        possible_times = {
            f"{hour_24}",
            f"{hour_12}",
            f"{hour_12} {start_datetime.strftime('%p').lower()}",
            f"{hour_12}{start_datetime.strftime('%p').lower()}",
            f"{hour_12}:{minute}",
            f"{hour_12}:{minute} {start_datetime.strftime('%p').lower()}",
            f"{hour_12}:{minute}{start_datetime.strftime('%p').lower()}",
        }

        if any(
            re.search(
                rf"\b{re.escape(time_text)}\b",
                normalized_message,
            )
            for time_text in possible_times
        ):
            return slot

    return None


def get_slot_selection_error(
    user_message: str,
    available_slots: list[dict[str, object]] | None,
) -> str | None:
    """Return a controlled clarification for an invalid slot choice."""

    if not available_slots:
        return None

    normalized_message = normalize_text(user_message)
    ambiguous_phrases = (
        "this slot",
        "this one",
        "that one",
        "take this",
        "choose this",
        "select this",
    )

    if (
        len(available_slots) > 1
        and any(
            phrase in normalized_message
            for phrase in ambiguous_phrases
        )
        and re.search(r"\b\d+\b", normalized_message) is None
    ):
        return (
            "Which slot do you mean? Please choose option 1, 2, "
            "or a time."
        )

    option_match = re.search(
        r"\b(?:option|slot)\s*(\d+)\b",
        normalized_message,
    )
    bare_number_match = re.fullmatch(r"\s*(\d+)\s*", user_message)
    selected_number = None

    if option_match is not None:
        selected_number = int(option_match.group(1))
    elif bare_number_match is not None:
        selected_number = int(bare_number_match.group(1))

    if (
        selected_number is not None
        and not 1 <= selected_number <= len(available_slots)
    ):
        option_range = (
            "option 1"
            if len(available_slots) == 1
            else (
                "options 1 and 2"
                if len(available_slots) == 2
                else f"options 1 to {len(available_slots)}"
            )
        )

        return (
            f"I only have {option_range} available. Would you like "
            "one of those, or a different date?"
        )

    return None


def extract_phone_candidate(
    user_message: str,
) -> str | None:
    """Extract a phone-like value, including invalid lengths."""

    phone_match = re.search(
        r"(?<![A-Za-z0-9])\+?\d[\d\s()\-]{6,}\d(?![A-Za-z0-9])",
        user_message,
    )

    if phone_match is None:
        return None

    candidate = phone_match.group(0).strip()
    digit_count = len(re.sub(r"\D", "", candidate))

    if digit_count < 9:
        return None

    return candidate


def normalize_sri_lankan_phone_number(
    phone_number: str,
) -> str | None:
    """Normalize a valid Sri Lankan mobile number to +94 format."""

    if re.search(r"[A-Za-z]", phone_number):
        return None

    compact_number = re.sub(
        r"[\s()\-]+",
        "",
        phone_number,
    )

    if compact_number.startswith("+94"):
        local_digits = compact_number[3:]
    elif compact_number.startswith("94"):
        local_digits = compact_number[2:]
    elif compact_number.startswith("0"):
        local_digits = compact_number[1:]
    else:
        local_digits = compact_number

    valid_mobile_prefixes = {
        "70",
        "71",
        "72",
        "75",
        "76",
        "77",
        "78",
    }

    if (
        len(local_digits) != 9
        or not local_digits.isdigit()
        or local_digits[:2] not in valid_mobile_prefixes
    ):
        return None

    if compact_number.startswith("+") and not compact_number.startswith(
        "+94",
    ):
        return None

    return f"+94{local_digits}"


def extract_phone_number(
    user_message: str,
) -> str | None:
    """Extract and normalize a valid Sri Lankan mobile number."""

    phone_candidate = extract_phone_candidate(user_message)

    if phone_candidate is None:
        return None

    return normalize_sri_lankan_phone_number(phone_candidate)


def extract_customer_name(
    user_message: str,
    phone_number: str | None,
) -> str | None:
    """Extract a simple customer name from a message."""

    cleaned_message = user_message.strip()

    stated_name_match = re.search(
        r"\b(?:my\s+name\s+is|name\s+is|i\s+am|i['’]m|this\s+is)"
        r"\s+([A-Za-z][A-Za-z .'-]{1,80}?)"
        r"(?=\s*[,;]|\s+(?:and\s+)?(?:my\s+)?(?:phone|mobile|"
        r"contact|number)\b|\s+\+?\d|$)",
        cleaned_message,
        flags=re.IGNORECASE,
    )

    if stated_name_match is None:
        stated_name_match = re.search(
            r"\bfor\s+([A-Za-z][A-Za-z .'-]{1,80}?)"
            r"(?=\s*[,;]\s*(?:phone|mobile|contact|number)\b)",
            cleaned_message,
            flags=re.IGNORECASE,
        )

    if stated_name_match is not None:
        stated_name = re.sub(
            r"\s+",
            " ",
            stated_name_match.group(1),
        ).strip(" .,-")

        if len(stated_name) >= 2:
            return stated_name.title()

    if phone_number is not None:
        cleaned_message = cleaned_message.replace(
            phone_number,
            "",
        )

        spaced_phone = " ".join(phone_number)
        cleaned_message = cleaned_message.replace(
            spaced_phone,
            "",
        )

    cleaned_message = re.sub(
        r"\b(?:my\s+name\s+is|name\s+is|i\s+am|"
        r"this\s+is|contact\s+number|phone\s+number|"
        r"mobile\s+number|number\s+is|contact)\b",
        "",
        cleaned_message,
        flags=re.IGNORECASE,
    )

    cleaned_message = re.sub(
        r"\b(?:and|is|my|phone|number)\b",
        " ",
        cleaned_message,
        flags=re.IGNORECASE,
    )

    cleaned_message = re.sub(
        r"[^A-Za-z .'-]+",
        " ",
        cleaned_message,
    )

    cleaned_message = re.sub(
        r"\s+",
        " ",
        cleaned_message,
    ).strip(" .,-")

    if len(cleaned_message) < 2:
        return None

    return cleaned_message.title()


def get_or_create_customer_from_details(
    full_name: str,
    phone_number: str,
) -> dict[str, object]:
    """Find an existing customer by phone or create a new one."""

    normalized_phone_number = normalize_sri_lankan_phone_number(
        phone_number,
    )

    if normalized_phone_number is None:
        raise ValueError("A valid Sri Lankan phone number is required.")

    database = SessionLocal()

    try:
        customer = (
            database.query(Customer)
            .filter(Customer.phone_number == normalized_phone_number)
            .first()
        )

        if customer is not None:
            customer.full_name = full_name
            customer.is_active = True
            database.commit()
            database.refresh(customer)

            return {
                "id": customer.id,
                "full_name": customer.full_name,
                "phone_number": customer.phone_number,
            }

        customer = Customer(
            full_name=full_name,
            phone_number=normalized_phone_number,
            is_active=True,
        )

        database.add(customer)
        database.commit()
        database.refresh(customer)

        return {
            "id": customer.id,
            "full_name": customer.full_name,
            "phone_number": customer.phone_number,
        }

    finally:
        database.close()


def find_selected_slot(
    state: AppointmentAgentState,
) -> dict[str, object] | None:
    """Find the selected slot from the saved available slots."""

    slot_id = state.get("slot_id")

    if slot_id is None:
        return None

    available_slots = state.get("available_slots") or []

    for slot in available_slots:
        if slot.get("slot_id") == slot_id:
            return slot

    return None


def format_booking_confirmation_summary(
    state: AppointmentAgentState,
) -> str:
    """Format the pending appointment before final confirmation."""

    selected_slot = find_selected_slot(state)

    service_name = state.get("service_name") or "Appointment"
    staff_name = state.get("staff_name") or "Selected staff"
    customer_name = state.get("customer_name") or "Customer"
    phone_number = state.get("customer_phone_number") or "Not provided"

    if selected_slot is not None:
        start_time = format_time(
            selected_slot.get("start_datetime"),
        )
        end_time = format_time(
            selected_slot.get("end_datetime"),
        )
    else:
        start_time = "Selected time"
        end_time = "Selected end time"

    friendly_date = format_date(
        state.get("requested_date"),
    )

    return (
        "Please confirm your appointment:\n\n"
        f"Service: {service_name}\n"
        f"Doctor/Staff: {staff_name}\n"
        f"Date: {friendly_date}\n"
        f"Time: {start_time} – {end_time}\n"
        f"Name: {customer_name}\n"
        f"Phone: {phone_number}\n\n"
        "Should I confirm this booking?"
    )


def format_reschedule_confirmation_summary(
    state: AppointmentAgentState,
) -> str:
    """Format the pending reschedule before final confirmation."""

    selected_slot = find_selected_slot(state)
    service_name = state.get("service_name") or "Appointment"
    staff_name = state.get("staff_name") or "Selected staff"

    if selected_slot is not None:
        start_time = format_time(
            selected_slot.get("start_datetime"),
        )
        end_time = format_time(
            selected_slot.get("end_datetime"),
        )
    else:
        start_time = "Selected time"
        end_time = "Selected end time"

    friendly_date = format_date(
        state.get("requested_date"),
    )

    return (
        "Please confirm your rescheduled appointment:\n\n"
        f"Appointment ID: {state['appointment_id']}\n"
        f"Service: {service_name}\n"
        f"Doctor/Staff: {staff_name}\n"
        f"New date: {friendly_date}\n"
        f"New time: {start_time} – {end_time}\n\n"
        "Should I confirm this reschedule?"
    )


def is_confirmation_yes_message(
    user_message: str,
) -> bool:
    """Return True when the user confirms a pending booking."""

    normalized_message = normalize_text(user_message)

    yes_phrases = {
        "yes",
        "ok",
        "okay",
        "sure",
        "yes confirm",
        "confirm",
        "confirm it",
        "book it",
        "okay confirm",
        "ok confirm",
        "please confirm",
        "go ahead",
        "yes book it",
    }

    return normalized_message in yes_phrases


def is_ambiguous_confirmation_message(
    user_message: str,
) -> bool:
    """Return True when confirmation wording is not decisive."""

    normalized_message = normalize_text(user_message)

    return normalized_message in {
        "maybe",
        "not sure",
        "i guess",
        "sure i guess",
    }


def is_confirmation_no_message(
    user_message: str,
) -> bool:
    """Return True when the user rejects a pending booking."""

    normalized_message = normalize_text(user_message)

    no_phrases = {
        "no",
        "no thanks",
        "dont confirm",
        "do not confirm",
        "not now",
        "cancel",
        "cancel it",
        "leave it",
        "keep it",
        "no wait keep it",
    }

    return normalized_message in no_phrases


def is_time_correction_message(
    user_message: str,
) -> bool:
    """Return True when the user wants to change the selected slot."""

    normalized_message = normalize_text(user_message)

    time_change_phrases = (
        "change time",
        "change the time",
        "chnage time",
        "chnage the time",
        "different time",
        "differnt time",
        "another time",
        "another slot",
        "different slot",
        "differnt slot",
        "change slot",
        "change the slot",
        "choose different time",
        "choose differnt time",
        "change appointment time",
        "change my time",
        "afternoon instead",
        "morning instead",
        "evening instead",
    )

    return any(
        phrase in normalized_message
        for phrase in time_change_phrases
    )


def is_date_correction_message(
    user_message: str,
) -> bool:
    """Return True when the user wants to change the appointment date."""

    normalized_message = normalize_text(user_message)

    date_change_phrases = (
        "change date",
        "change the date",
        "different date",
        "differnt date",
        "another date",
        "choose different date",
        "choose differnt date",
        "change appointment date",
    )

    if any(
        phrase in normalized_message
        for phrase in date_change_phrases
    ):
        return True

    return (
        any(phrase in normalized_message for phrase in ("i meant", "no i meant"))
        and parse_requested_date(user_message) is not None
    )


def is_service_correction_message(
    user_message: str,
) -> bool:
    """Return True when the user wants to change the service."""

    normalized_message = normalize_text(user_message)

    service_change_phrases = (
        "change service",
        "change the service",
        "different service",
        "differnt service",
        "another service",
        "choose different service",
        "choose differnt service",
        "change appointment service",
    )

    if any(
        phrase in normalized_message
        for phrase in service_change_phrases
    ):
        return True

    service_words = (
        "dental",
        "dentist",
        "physiotherapy",
        "physio",
        "dermatology",
        "skin",
        "general consultation",
    )
    return (
        any(marker in normalized_message for marker in ("not ", "i meant", "instead"))
        and any(word in normalized_message for word in service_words)
    )


def is_targeted_service_correction_message(user_message: str) -> bool:
    """Return True when a correction states the replacement service."""

    normalized_message = normalize_text(user_message)
    service_words = (
        "dental",
        "dentist",
        "physiotherapy",
        "physio",
        "dermatology",
        "skin",
        "general consultation",
    )
    return (
        is_service_correction_message(user_message)
        and any(word in normalized_message for word in service_words)
    )


def is_staff_correction_message(user_message: str) -> bool:
    """Return True when the selected staff member should be changed."""

    normalized_message = normalize_text(user_message)
    return any(
        phrase in normalized_message
        for phrase in (
            "change doctor",
            "change the doctor",
            "change only the doctor",
            "different doctor",
            "another doctor",
            "different staff",
            "another staff",
            "with dr ",
            "with doctor ",
        )
    )


def is_phone_correction_message(user_message: str) -> bool:
    """Return True when a supplied phone number replaces the prior number."""

    normalized_message = normalize_text(user_message)
    return any(
        phrase in normalized_message
        for phrase in (
            "other number",
            "new number",
            "correct number",
            "number is actually",
            "use this number",
        )
    )


def is_name_correction_message(user_message: str) -> bool:
    """Return True when the customer explicitly corrects their name."""

    normalized_message = normalize_text(user_message)
    return any(
        phrase in normalized_message
        for phrase in (
            "sorry my name is",
            "correct name is",
            "name is actually",
            "i meant my name",
        )
    )


def analyze_mixed_intents(
    user_message: str,
) -> tuple[str | None, list[str], str | None]:
    """Identify safe secondary goals and clarify conflicting mutations."""

    normalized_message = normalize_text(user_message)
    mutations: list[str] = []

    if any(
        phrase in normalized_message
        for phrase in (
            "book a",
            "book dental",
            "book physiotherapy",
            "book dermatology",
            "book new",
            "new appointment",
            "make an appointment",
            "get me an appointment",
        )
    ):
        mutations.append("book_appointment")
    if any(
        phrase in normalized_message
        for phrase in (
            "cancel my",
            "cancel the",
            "cancel dental",
            "cancel physiotherapy",
            "remove my appointment",
            "wont be able to make it",
            "won t be able to make it",
        )
    ):
        mutations.append("cancel_appointment")
    if any(
        phrase in normalized_message
        for phrase in (
            "reschedule",
            "move my appointment",
            "shift my booking",
            "change my appointment",
        )
    ):
        mutations.append("reschedule_appointment")

    unique_mutations = list(dict.fromkeys(mutations))
    if len(unique_mutations) > 1:
        readable_actions = [
            intent.replace("_appointment", "").replace("_", " ")
            for intent in unique_mutations
        ]
        return None, unique_mutations, (
            "I can help with both, but I can only handle one change at "
            f"a time. Should we {readable_actions[0]} first or "
            f"{readable_actions[1]} first?"
        )

    primary_intent = unique_mutations[0] if unique_mutations else None
    secondary_intents: list[str] = []

    if is_pricing_request(user_message):
        secondary_intents.append("ask_pricing")
    if "how long" in normalized_message or "duration" in normalized_message:
        secondary_intents.append("ask_duration")
    if "cancellation policy" in normalized_message:
        secondary_intents.append("ask_cancellation_policy")

    if is_appointment_listing_request(user_message):
        secondary_intents.append("view_appointments")

    has_availability_goal = (
        any(
            word in normalized_message
            for word in ("available", "availability", "free slot")
        )
        and parse_requested_date(user_message) is not None
    )
    if has_availability_goal:
        if primary_intent is None:
            primary_intent = "check_availability"
        elif primary_intent != "check_availability":
            secondary_intents.append("check_availability")

    secondary_intents = [
        intent
        for intent in dict.fromkeys(secondary_intents)
        if intent != primary_intent
    ]
    return primary_intent, secondary_intents, None


def is_thank_you_message(
    user_message: str,
) -> bool:
    """Return True when the user is closing politely after help."""

    normalized_message = normalize_text(user_message)

    thank_you_phrases = (
        "thank you",
        "thanks",
        "thank u",
        "oki thank you",
        "ok thank you",
        "okay thank you",
        "oki thanks",
        "ok thanks",
        "okay thanks",
        "thats all",
        "that is all",
        "all good",
    )

    return any(
        phrase in normalized_message
        for phrase in thank_you_phrases
    )


def is_human_handoff_message(
    user_message: str,
) -> bool:
    """Return True when the user asks to continue with a person."""

    normalized_message = normalize_text(user_message)
    handoff_phrases = (
        "transfer to human",
        "talk to human",
        "talk to a human",
        "talk to a person",
        "speak to human",
        "speak to a human",
        "speak to a person",
        "human please",
        "front desk",
        "receptionist",
        "staff member",
        "connect me to someone",
    )

    return (
        normalized_message in {"human", "staff"}
        or any(
            phrase in normalized_message
            for phrase in handoff_phrases
        )
    )


def is_graceful_exit_message(
    user_message: str,
) -> bool:
    """Return True when the user politely stops the current request."""

    normalized_message = normalize_text(user_message)
    exit_phrases = (
        "thank you",
        "thanks",
        "sorry",
        "no problem",
        "leave it",
        "cancel this request",
        "never mind",
        "forget it",
        "stop",
    )

    return any(
        phrase in normalized_message
        for phrase in exit_phrases
    )


def format_graceful_exit_response(
    user_message: str,
) -> str:
    """Format a polite controlled response that exits the active flow."""

    if is_thank_you_message(user_message):
        return (
            "You're welcome. Let me know if you need help "
            "with anything else."
        )

    normalized_message = normalize_text(user_message)

    if "sorry" in normalized_message or "no problem" in normalized_message:
        return (
            "No problem. Let me know if you need help with "
            "anything else."
        )

    return (
        "No problem, I've stopped that request. Let me know if you "
        "need help with anything else."
    )


def is_explicit_new_booking_message(
    user_message: str,
) -> bool:
    """Return True only for an explicit request to leave rescheduling."""

    normalized_message = normalize_text(user_message)
    new_booking_phrases = (
        "book another appointment",
        "book new appointment",
        "new appointment",
        "start new booking",
    )

    return any(
        phrase in normalized_message
        for phrase in new_booking_phrases
    )


def new_booking_reset_updates() -> AppointmentAgentState:
    """Clear every transactional value before a separate new booking."""

    return {
        "intent": "book_appointment",
        "service_id": None,
        "service_name": None,
        "staff_id": None,
        "staff_name": None,
        "requested_date": None,
        "time_preference": None,
        "time_preference_error": None,
        "available_slots": None,
        "upcoming_alternative_slots": None,
        "selected_slot_summary": None,
        "booking_summary": None,
        "slot_id": None,
        "appointment_id": None,
        "appointment_reference_number": None,
        "appointment_status": None,
        "current_slot_id": None,
        "cancellation_reason": None,
        "missing_fields": [],
        "pending_action_started_at": None,
        "slot_options_updated_at": None,
        "state_expired_message": None,
        "semantic_nlu": None,
        "secondary_intents": [],
        "mixed_intent_clarification": None,
        "date_clarification": None,
        "clarification_attempts": 0,
        "slot_selection_error": None,
        "tool_error": None,
        "next_question": None,
        "confirmation_status": "not_requested",
        "transaction_started_explicitly": True,
        "paused_intent": None,
    }


def switch_transaction_updates(
    state: AppointmentAgentState,
    new_intent: AppointmentIntent,
) -> AppointmentAgentState:
    """Switch explicit actions without carrying unsafe pending choices."""

    if new_intent == "book_appointment":
        return new_booking_reset_updates()

    preserve_appointment = state.get("appointment_id") is not None
    return {
        "intent": new_intent,
        "appointment_id": state.get("appointment_id"),
        "appointment_reference_number": state.get(
            "appointment_reference_number"
        ),
        "appointment_status": state.get("appointment_status"),
        "current_slot_id": state.get("current_slot_id"),
        "service_id": state.get("service_id") if preserve_appointment else None,
        "service_name": state.get("service_name") if preserve_appointment else None,
        "staff_id": state.get("staff_id") if preserve_appointment else None,
        "staff_name": state.get("staff_name") if preserve_appointment else None,
        "requested_date": None,
        "time_preference": None,
        "time_preference_error": None,
        "available_slots": None,
        "upcoming_alternative_slots": None,
        "slot_id": None,
        "selected_slot_summary": None,
        "booking_summary": None,
        "confirmation_status": "not_requested",
        "pending_action_started_at": None,
        "slot_options_updated_at": None,
        "slot_selection_error": None,
        "tool_error": None,
        "next_question": None,
        "transaction_started_explicitly": True,
        "paused_intent": None,
    }


def is_upcoming_availability_request(
    user_message: str,
) -> bool:
    """Return True when the user asks for other or upcoming availability."""

    normalized_message = normalize_text(user_message)

    upcoming_availability_phrases = (
        "which date available",
        "which dates are available",
        "tell me available dates",
        "show available dates",
        "any available date",
        "when are you available",
        "other available slots",
        "any other available slots",
        "other slots",
        "any other slots",
        "show me other slots",
        "next available slot",
        "next available slots",
    )

    return (
        any(
            phrase in normalized_message
            for phrase in upcoming_availability_phrases
        )
        or (
            "other" in normalized_message
            and "slot" in normalized_message
            and "available" in normalized_message
        )
    )


def is_appointment_listing_request(
    user_message: str,
) -> bool:
    """Return True for appointment listing or status requests."""

    normalized_message = normalize_text(user_message)
    listing_phrases = (
        "all my appointments",
        "tell me all my appointments",
        "show my appointments",
        "list my appointments",
        "my appointments",
        "do i have any appointments",
        "appointment status",
        "check my appointment",
        "is my appointment confirmed",
        "what appointments do i have",
        "which appointments do i have",
        "show my bookings",
        "my bookings",
    )

    return any(
        phrase in normalized_message
        for phrase in listing_phrases
    )


def is_capabilities_request(user_message: str) -> bool:
    """Return True when the user asks what the assistant can do."""

    normalized_message = normalize_text(user_message)

    return any(
        phrase in normalized_message
        for phrase in (
            "what can you do",
            "how can you help",
            "help me",
        )
    )


def is_greeting_or_small_talk(user_message: str) -> bool:
    """Return True for greetings and basic assistant small talk."""

    normalized_message = normalize_text(user_message)

    return normalized_message in {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "how r u",
    }


def is_safe_information_request(user_message: str) -> bool:
    """Return True for clinic facts that require verified knowledge."""

    normalized_message = normalize_text(user_message)

    return any(
        phrase in normalized_message
        for phrase in (
            "opening hours",
            "cancellation policy",
            "where are you located",
            "your location",
            "accept insurance",
            "insurance",
        )
    )


def is_pricing_request(user_message: str) -> bool:
    """Return True when the user asks for a service price."""

    normalized_message = normalize_text(user_message)

    return any(
        phrase in normalized_message
        for phrase in (
            "how much",
            "price of",
            "cost of",
            "price for",
        )
    )


def is_likely_gibberish(user_message: str) -> bool:
    """Detect short inputs with no recognizable language structure."""

    normalized_message = normalize_text(user_message)

    if not normalized_message:
        return False

    words = normalized_message.split()

    return len(words) == 1 and len(words[0]) >= 8


def find_upcoming_available_slots(
    service_id: int,
    staff_id: int | None = None,
    days_to_search: int = 14,
) -> list[dict[str, object]]:
    """Search real availability over the next several days."""

    upcoming_slots: list[dict[str, object]] = []
    today = business_today()

    for day_offset in range(days_to_search):
        requested_date = (
            today + timedelta(days=day_offset)
        ).isoformat()
        tool_arguments: dict[str, object] = {
            "service_id": service_id,
            "requested_date": requested_date,
        }

        if staff_id is not None:
            tool_arguments["staff_id"] = staff_id

        slots = get_validated_available_slots(tool_arguments)
        upcoming_slots.extend(slots)

    return sorted(
        upcoming_slots,
        key=lambda slot: str(slot.get("start_datetime") or ""),
    )


def validate_available_slot_results(
    result: object,
    service_id: int,
    staff_id: int | None = None,
) -> list[dict[str, object]]:
    """Validate availability-tool output before it reaches conversation text."""

    if not isinstance(result, list):
        raise ValueError("Availability returned an invalid result.")

    required_fields = {
        "slot_id",
        "service_id",
        "staff_id",
        "start_datetime",
        "end_datetime",
    }
    validated: list[dict[str, object]] = []

    for slot in result:
        if not isinstance(slot, dict) or not required_fields.issubset(slot):
            raise ValueError("Availability returned an incomplete slot.")

        try:
            slot_service_id = int(slot["service_id"])
            slot_staff_id = int(slot["staff_id"])
            int(slot["slot_id"])
        except (TypeError, ValueError):
            raise ValueError("Availability returned invalid identifiers.")

        if slot_service_id != service_id:
            raise ValueError("Availability returned a slot for another service.")

        if staff_id is not None and slot_staff_id != staff_id:
            raise ValueError("Availability returned a slot for another staff member.")

        status = slot.get("status")
        if status is not None and str(status).upper() != "AVAILABLE":
            raise ValueError("Availability returned a slot that is not available.")

        if (
            parse_datetime(slot.get("start_datetime")) is None
            or parse_datetime(slot.get("end_datetime")) is None
        ):
            raise ValueError("Availability returned an invalid time range.")

        validated.append(slot)

    return validated


def get_validated_available_slots(
    tool_arguments: dict[str, object],
) -> list[dict[str, object]]:
    """Call availability once and enforce its read-only output contract."""

    raw_result = check_available_slots.run(tool_arguments)
    return validate_available_slot_results(
        raw_result,
        service_id=int(tool_arguments["service_id"]),
        staff_id=(
            int(tool_arguments["staff_id"])
            if tool_arguments.get("staff_id") is not None
            else None
        ),
    )

def create_confirmed_appointment_from_state(
    state: AppointmentAgentState,
) -> dict[str, object]:
    """Create a confirmed appointment using resolved state values."""

    database = SessionLocal()

    try:
        appointment = create_appointment(
            database=database,
            appointment_data=AppointmentCreate(
                customer_id=int(state["customer_id"]),
                service_id=int(state["service_id"]),
                staff_id=int(state["staff_id"]),
                slot_id=int(state["slot_id"]),
                customer_notes=None,
            ),
        )

        return {
            "id": appointment.id,
            "reference_number": appointment.reference_number,
            "slot_id": appointment.slot_id,
            "start_datetime": appointment.start_datetime.isoformat(),
            "end_datetime": appointment.end_datetime.isoformat(),
        }

    finally:
        database.close()

def reschedule_confirmed_appointment_from_state(
    state: AppointmentAgentState,
) -> dict[str, object]:
    """Reschedule an existing appointment using resolved state values."""

    database = SessionLocal()

    try:
        appointment = reschedule_appointment(
            database=database,
            appointment_id=int(state["appointment_id"]),
            new_slot_id=int(state["slot_id"]),
        )

        return {
            "id": appointment.id,
            "reference_number": appointment.reference_number,
            "slot_id": appointment.slot_id,
            "start_datetime": appointment.start_datetime.isoformat(),
            "end_datetime": appointment.end_datetime.isoformat(),
        }

    finally:
        database.close()


def cancel_confirmed_appointment_from_state(
    state: AppointmentAgentState,
) -> dict[str, object]:
    """Cancel a confirmed appointment after explicit confirmation."""

    database = SessionLocal()

    try:
        appointment = cancel_appointment(
            database=database,
            appointment_id=int(state["appointment_id"]),
            cancellation_reason=(
                state.get("cancellation_reason")
                or "Cancelled by customer through AI assistant"
            ),
        )

        return appointment_to_conversation_details(appointment)

    finally:
        database.close()


def format_confirmed_appointment_response(
    state: AppointmentAgentState,
    appointment: dict[str, object],
) -> str:
    """Format the final confirmed appointment response."""

    selected_slot = find_selected_slot(state)

    service_name = state.get("service_name") or "Appointment"
    staff_name = state.get("staff_name") or "Selected staff"

    if selected_slot is not None:
        start_time = format_time(
            selected_slot.get("start_datetime"),
        )
        end_time = format_time(
            selected_slot.get("end_datetime"),
        )
    else:
        start_time = format_time(
            appointment.get("start_datetime"),
        )
        end_time = format_time(
            appointment.get("end_datetime"),
        )

    friendly_date = format_date(
        state.get("requested_date"),
    )

    return (
        "Your appointment is confirmed.\n\n"
        f"Reference: {appointment['reference_number']}\n"
        f"Service: {service_name}\n"
        f"Doctor/Staff: {staff_name}\n"
        f"Date: {friendly_date}\n"
        f"Time: {start_time} – {end_time}"
    )


def format_rescheduled_appointment_response(
    state: AppointmentAgentState,
    appointment: dict[str, object],
) -> str:
    """Format the final rescheduled appointment response."""

    selected_slot = find_selected_slot(state)
    service_name = state.get("service_name") or "Appointment"
    staff_name = state.get("staff_name") or "Selected staff"

    if selected_slot is not None:
        start_time = format_time(
            selected_slot.get("start_datetime"),
        )
        end_time = format_time(
            selected_slot.get("end_datetime"),
        )
    else:
        start_time = format_time(
            appointment.get("start_datetime"),
        )
        end_time = format_time(
            appointment.get("end_datetime"),
        )

    friendly_date = format_date(
        state.get("requested_date"),
    )

    return (
        "Your appointment has been rescheduled.\n\n"
        f"Reference: {appointment['reference_number']}\n"
        f"Service: {service_name}\n"
        f"Doctor/Staff: {staff_name}\n"
        f"New date: {friendly_date}\n"
        f"New time: {start_time} – {end_time}"
    )


def detect_intent(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Identify the user's current appointment intent."""

    raw_user_message = get_latest_user_message(state)
    user_message = normalize_domain_typos(raw_user_message).lower()

    nlu_result, semantic_nlu = classify_message_for_state(
        state,
        raw_user_message,
    )

    current_intent = state.get("intent")
    mutation_intents = {
        "book_appointment",
        "cancel_appointment",
        "reschedule_appointment",
    }

    if (
        is_explicit_new_booking_message(user_message)
        and (
            state.get("appointment_id") is not None
            or state.get("confirmation_status") in {"confirmed", "rejected"}
        )
    ):
        return new_booking_reset_updates()

    if (
        state.get("confirmation_status") == "confirmed"
        and is_confirmation_yes_message(user_message)
    ):
        reference = state.get("appointment_reference_number")
        reference_text = f" Reference: {reference}." if reference else ""
        completed_messages = {
            "book_appointment": "That appointment is already confirmed.",
            "cancel_appointment": "That appointment is already cancelled.",
            "reschedule_appointment": "That appointment has already been rescheduled.",
        }
        return {
            "next_question": completed_messages.get(
                str(current_intent),
                "That request is already complete.",
            ) + reference_text,
        }

    if (
        current_intent in mutation_intents
        and nlu_result.intent in mutation_intents
        and nlu_result.intent != current_intent
        and not (
            current_intent == "book_appointment"
            and (
                is_time_correction_message(user_message)
                or is_date_correction_message(user_message)
                or is_service_correction_message(user_message)
            )
        )
        and not (
            nlu_result.intent == "book_appointment"
            and not is_explicit_new_booking_message(user_message)
        )
    ):
        return switch_transaction_updates(
            state,
            nlu_result.intent,  # type: ignore[arg-type]
        )

    if (
        current_intent in mutation_intents
        and nlu_result.intent == "view_appointments"
    ):
        return {
            "intent": "view_appointments",
            "paused_intent": current_intent,
            "next_question": None,
            "transaction_started_explicitly": True,
        }

    if (
        state.get("state_expired_message") is not None
        and nlu_result.intent in {
            "book_appointment",
            "cancel_appointment",
            "reschedule_appointment",
            "check_availability",
            "view_appointments",
        }
    ):
        return {
            "intent": nlu_result.intent,
            "state_expired_message": None,
            "next_question": None,
            "transaction_started_explicitly": True,
        }
        # When availability has already been displayed, selecting an option
    # should continue into the booking flow.
    if (
        current_intent == "check_availability"
        and state.get("available_slots")
        and find_slot_from_message(
            raw_user_message,
            state.get("available_slots"),
        )
        is not None
    ):
        return {
            "intent": "book_appointment",
            "booking_summary": None,
            "confirmation_status": "not_requested",
            "next_question": None,
            "transaction_started_explicitly": True,
        }

    mixed_primary, secondary_intents, mixed_clarification = (
        analyze_mixed_intents(raw_user_message)
    )
    if mixed_clarification is not None:
        return {
            "intent": "general_question",
            "secondary_intents": secondary_intents,
            "mixed_intent_clarification": mixed_clarification,
            "next_question": mixed_clarification,
            "booking_summary": None,
            "confirmation_status": "not_requested",
        }
    if mixed_primary == "check_availability":
        return {
            "intent": "check_availability",
            "available_slots": None,
            "upcoming_alternative_slots": None,
            "slot_id": None,
            "selected_slot_summary": None,
            "booking_summary": None,
            "confirmation_status": "not_requested",
            "secondary_intents": secondary_intents,
            "mixed_intent_clarification": None,
            "next_question": None,
            "transaction_started_explicitly": True,
        }
    if mixed_primary is not None and secondary_intents:
        return {
            "intent": mixed_primary,
            "secondary_intents": secondary_intents,
            "mixed_intent_clarification": None,
            "confirmation_status": "not_requested",
            "next_question": None,
        }

    if semantic_nlu is not None:
        if nlu_result.requires_clarification:
            return {
                "intent": state.get("intent") or "general_question",
                "semantic_nlu": semantic_nlu,
                "next_question": (
                    nlu_result.clarification_reason
                    or "Could you clarify what you would like me to do?"
                ),
            }

        if nlu_result.intent in INFORMATION_INTERRUPTION_INTENTS:
            return {
                "intent": informational_response_intent(state),
                "semantic_nlu": semantic_nlu,
                "next_question": None,
            }

        if nlu_result.intent in {
            "book_appointment",
            "cancel_appointment",
            "reschedule_appointment",
            "check_availability",
            "view_appointments",
            "list_services",
        }:
            current_intent = state.get("intent")
            if (
                current_intent in ACTIVE_FLOW_INTENTS
                and nlu_result.intent in ACTIVE_FLOW_INTENTS
                and nlu_result.intent != current_intent
            ):
                return {
                    "intent": current_intent,
                    "semantic_nlu": semantic_nlu,
                    "next_question": (
                        f"You are currently working on {str(current_intent).replace('_', ' ')}. "
                        f"Should I stop that and switch to "
                        f"{nlu_result.intent.replace('_', ' ')}?"
                    ),
                }

            return {
                "intent": nlu_result.intent,
                "semantic_nlu": semantic_nlu,
                "next_question": None,
                "transaction_started_explicitly": (
                    nlu_result.intent in ACTIVE_FLOW_INTENTS
                ),
            }

        return {
            "intent": "general_question",
            "semantic_nlu": semantic_nlu,
            "next_question": None,
        }

    if (
        state.get("intent") in ACTIVE_FLOW_INTENTS
        and nlu_result.intent in INFORMATION_INTERRUPTION_INTENTS
    ):
        return {
            "intent": state["intent"],
            "next_question": None,
        }

    if (
        state.get("intent") == "check_availability"
        and normalize_text(user_message) in {
            "when is available",
            "when available",
            "what is available",
        }
    ):
        return {
            "intent": "check_availability",
            "requested_date": None,
            "available_slots": None,
            "upcoming_alternative_slots": None,
            "slot_id": None,
            "selected_slot_summary": None,
            "next_question": None,
            "transaction_started_explicitly": True,
        }

    active_transaction_intent = state.get("intent") in {
        "book_appointment",
        "reschedule_appointment",
        "cancel_appointment",
    }

    if (
        not active_transaction_intent
        and nlu_result.intent in {
            "book_appointment",
            "reschedule_appointment",
            "cancel_appointment",
        }
    ):
        return {
            "intent": nlu_result.intent,
            "transaction_started_explicitly": True,
        }
    if (
        not active_transaction_intent
        and nlu_result.intent in {
            "check_availability",
            "view_appointments",
        }
    ):
        return {
            "intent": nlu_result.intent,
            "next_question": None,
            "transaction_started_explicitly": True,
        }

    if nlu_result.intent == "ask_duration":
        return {
            "intent": "general_question",
            "next_question": None,
        }

    if is_human_handoff_message(user_message):
        return {
            "intent": "general_question",
            "next_question": (
                "I can hand this over to a human staff member. "
                "Please contact the front desk or clinic staff to "
                "continue this request."
            ),
            "state_expired_message": None,
            "booking_summary": None,
            "available_slots": None,
            "slot_id": None,
            "selected_slot_summary": None,
            "confirmation_status": "not_requested",
            "transaction_started_explicitly": False,
            "paused_intent": None,
        }

    if is_graceful_exit_message(user_message):
        return {
            "intent": "general_question",
            "next_question": format_graceful_exit_response(
                user_message,
            ),
            "booking_summary": None,
            "requested_date": None,
            "available_slots": None,
            "slot_id": None,
            "selected_slot_summary": None,
            "confirmation_status": "not_requested",
            "transaction_started_explicitly": False,
            "paused_intent": None,
        }
    if (
        state.get("intent") == "view_appointments"
        and state.get("confirmation_status") != "pending"
        and normalize_text(user_message) in {
            "ok",
            "okay",
            "alright",
            "got it",
        }
    ):
        return {
            "intent": "general_question",
            "next_question": (
                "Okay. Let me know if you need help with anything else."
            ),
            "paused_intent": None,
            "transaction_started_explicitly": False,
        }

    if is_appointment_listing_request(user_message):
        return {
            "intent": "view_appointments",
            "next_question": None,
        }

    if is_appointment_listing_request(user_message):
        return {
            "intent": "view_appointments",
            "next_question": None,
        }

    if (
        is_capabilities_request(user_message)
        or is_greeting_or_small_talk(user_message)
        or is_safe_information_request(user_message)
        or is_pricing_request(user_message)
    ):
        return {
            "intent": "general_question",
            "next_question": None,
        }

    if (
        state.get("appointment_id") is not None
        and is_thank_you_message(user_message)
    ):
        return {
            "intent": "general_question",
            "next_question": (
                "You're welcome. Let me know if you need help "
                "with anything else."
            ),
            "confirmation_status": "confirmed",
        }

    if (
        state.get("intent") in {
            "book_appointment",
            "reschedule_appointment",
        }
        and is_staff_correction_message(user_message)
    ):
        return {
            "intent": state["intent"],
            "staff_id": None,
            "staff_name": None,
            "available_slots": None,
            "upcoming_alternative_slots": None,
            "slot_id": None,
            "selected_slot_summary": None,
            "booking_summary": None,
            "confirmation_status": "not_requested",
            "next_question": None,
        }

    if (
        state.get("intent") == "reschedule_appointment"
        and state.get("confirmation_status") == "pending"
        and state.get("booking_summary") is not None
    ):
        if is_confirmation_yes_message(user_message):
            return {
                "intent": "reschedule_appointment",
                "confirmation_status": "confirmed",
            }

        if is_confirmation_no_message(user_message):
            return {
                "intent": "reschedule_appointment",
                "confirmation_status": "rejected",
            }

    if (
        state.get("intent") == "cancel_appointment"
        and state.get("confirmation_status") == "pending"
        and state.get("booking_summary") is not None
    ):
        if is_confirmation_yes_message(user_message):
            return {
                "intent": "cancel_appointment",
                "confirmation_status": "confirmed",
            }

        if is_confirmation_no_message(user_message):
            return {
                "intent": "cancel_appointment",
                "confirmation_status": "rejected",
            }

    if (
        state.get("intent") == "reschedule_appointment"
        and is_date_correction_message(user_message)
    ):
        return {
            "intent": "reschedule_appointment",
            "requested_date": None,
            "available_slots": None,
            "slot_id": None,
            "selected_slot_summary": None,
            "booking_summary": None,
            "confirmation_status": "not_requested",
            "next_question": None,
        }

    is_reschedule_request = any(
        keyword in user_message
        for keyword in (
            "reschedule",
            "reshedule",
            "move my appointment",
            "change my appointment",
            "change the appointment",
        )
    )

    if (
        state.get("intent") == "reschedule_appointment"
        and is_reschedule_request
    ):
        return {
            "intent": "reschedule_appointment",
        }

    if is_reschedule_request:
        return {
            "intent": "reschedule_appointment",
            "requested_date": None,
            "slot_id": None,
            "selected_slot_summary": None,
            "booking_summary": None,
            "available_slots": None,
            "confirmation_status": "not_requested",
            "next_question": None,
            "transaction_started_explicitly": True,
        }

    normalized_user_message = normalize_text(user_message)

    if any(
        phrase in normalized_user_message
        for phrase in (
            "cancel",
            "remove my appointment",
            "cant come",
            "can t come",
            "cannot come",
            "cant make it",
            "can t make it",
            "cannot make it",
            "wont be able to make it",
            "won t be able to make it",
            "wont be able to come",
            "won t be able to come",
            "unable to attend",
        )
    ):
        return {
            "intent": "cancel_appointment",
            "booking_summary": None,
            "confirmation_status": "not_requested",
            "next_question": None,
            "transaction_started_explicitly": True,
        }

    if state.get("intent") == "reschedule_appointment":
        if is_explicit_new_booking_message(user_message):
            return new_booking_reset_updates()

        return {
            "intent": "reschedule_appointment",
        }

    is_new_booking_request = any(
        keyword in user_message
        for keyword in (
            "book another",
            "another appointment",
            "new appointment",
            "book new",
            "book physiotherapy",
            "book dental",
            "book dermatology",
            "book general",
            "book",
            "make an appointment",
            "need an appointment",
            "need a",
            "schedule an appointment",
            "see a doctor",
            "see the doctor",
        )
    )

    if (
        state.get("appointment_id") is not None
        and is_new_booking_request
    ):
        return new_booking_reset_updates()

    if (
        state.get("confirmation_status") == "pending"
        and state.get("appointment_id") is None
        and state.get("booking_summary") is not None
    ):
        if is_time_correction_message(user_message):
            return {
                "intent": "book_appointment",
                "slot_id": None,
                "selected_slot_summary": None,
                "booking_summary": None,
                "appointment_id": None,
                "appointment_reference_number": None,
                "confirmation_status": "not_requested",
            }

        if is_date_correction_message(user_message):
            return {
                "intent": "book_appointment",
                "requested_date": None,
                "available_slots": None,
                "slot_id": None,
                "selected_slot_summary": None,
                "booking_summary": None,
                "appointment_id": None,
                "appointment_reference_number": None,
                "confirmation_status": "not_requested",
            }

        if (
            is_service_correction_message(user_message)
            and not is_targeted_service_correction_message(user_message)
        ):
            return {
                "intent": "book_appointment",
                "service_id": None,
                "service_name": None,
                "staff_id": None,
                "staff_name": None,
                "requested_date": None,
                "available_slots": None,
                "slot_id": None,
                "selected_slot_summary": None,
                "booking_summary": None,
                "appointment_id": None,
                "appointment_reference_number": None,
                "confirmation_status": "not_requested",
            }

        if is_confirmation_yes_message(user_message):
            return {
                "intent": "book_appointment",
                "confirmation_status": "confirmed",
            }

        if is_confirmation_no_message(user_message):
            return {
                "intent": "book_appointment",
                "confirmation_status": "rejected",
            }

    if (
        state.get("confirmation_status") == "rejected"
        and state.get("appointment_id") is None
    ):
        if is_time_correction_message(user_message):
            return {
                "intent": "book_appointment",
                "slot_id": None,
                "selected_slot_summary": None,
                "booking_summary": None,
                "appointment_id": None,
                "appointment_reference_number": None,
                "confirmation_status": "not_requested",
            }

        if is_date_correction_message(user_message):
            return {
                "intent": "book_appointment",
                "requested_date": None,
                "available_slots": None,
                "slot_id": None,
                "selected_slot_summary": None,
                "booking_summary": None,
                "appointment_id": None,
                "appointment_reference_number": None,
                "confirmation_status": "not_requested",
            }

        if (
            is_service_correction_message(user_message)
            and not is_targeted_service_correction_message(user_message)
        ):
            return {
                "intent": "book_appointment",
                "service_id": None,
                "service_name": None,
                "staff_id": None,
                "staff_name": None,
                "requested_date": None,
                "available_slots": None,
                "slot_id": None,
                "selected_slot_summary": None,
                "booking_summary": None,
                "appointment_id": None,
                "appointment_reference_number": None,
                "confirmation_status": "not_requested",
            }

    if any(
        keyword in user_message
        for keyword in (
            "available slot",
            "available slots",
            "available time",
            "availability",
            "avalable",
            "free slot",
            "what time",
            "which time",
        )
    ):
        intent: AppointmentIntent = "check_availability"

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
            "appointment",
            "appoinment",
            "appointmentt",
            "appoinmtent",
            "make an appointment",
            "need an appointment",
            "need a",
            "schedule an appointment",
            "see a doctor",
            "see the doctor",
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
            "view_appointments",
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
    entity_message = entity_source_message(state, user_message)
    extracted: AppointmentAgentState = {}

    phone_candidate_for_turn = extract_phone_candidate(user_message)
    selected_slot = (
        None
        if phone_candidate_for_turn is not None
        else find_slot_from_message(
            user_message=user_message,
            available_slots=state.get("available_slots"),
        )
    )

    slot_selection_error = (
        None
        if phone_candidate_for_turn is not None
        else get_slot_selection_error(
            user_message=user_message,
            available_slots=state.get("available_slots"),
        )
    )

    if slot_selection_error is not None:
        extracted["slot_selection_error"] = slot_selection_error

    if selected_slot is not None:
        extracted["slot_id"] = int(selected_slot["slot_id"])
        extracted["staff_id"] = int(selected_slot["staff_id"])
        extracted["staff_name"] = (
            str(selected_slot["staff_name"])
            if selected_slot.get("staff_name") is not None
            else None
        )
        extracted["selected_slot_summary"] = format_slot_summary(
            selected_slot,
        )
        extracted["upcoming_alternative_slots"] = None

        if state.get("slot_selection_error") is not None:
            extracted["slot_selection_error"] = None

        selected_start = parse_datetime(
            selected_slot.get("start_datetime"),
        )

        if selected_start is not None:
            extracted["requested_date"] = (
                selected_start.date().isoformat()
            )

        if (
            state.get("confirmation_status") == "pending"
            and state.get("intent") == "book_appointment"
        ):
            extracted["confirmation_status"] = "not_requested"
            extracted["booking_summary"] = None
            extracted["appointment_id"] = None
            extracted["appointment_reference_number"] = None

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
        if field_name == "slot_id" and state.get("available_slots"):
            continue

        match = re.search(
            pattern,
            user_message,
            flags=re.IGNORECASE,
        )

        if match is not None:
            extracted[field_name] = int(match.group(1))

    if (
        state.get("intent") in {
            "reschedule_appointment",
            "cancel_appointment",
        }
        and state.get("appointment_id") is None
    ):
        numeric_id_match = re.fullmatch(
            r"\s*(\d+)\s*",
            user_message,
        )

        if numeric_id_match is not None:
            extracted["appointment_id"] = int(
                numeric_id_match.group(1),
            )

    if state.get("intent") in {
        "reschedule_appointment",
        "cancel_appointment",
        "view_appointments",
    }:
        reference_number = extract_appointment_reference(
            user_message,
        )

        if reference_number is not None:
            extracted["appointment_reference_number"] = (
                reference_number
            )
            appointment_details = get_appointment_by_reference(
                reference_number,
            )

            if appointment_details is not None:
                extracted.update(appointment_details)

    parsed_requested_date = None
    date_clarification = get_date_clarification_message(entity_message)

    if date_clarification is not None:
        extracted["date_clarification"] = date_clarification
        extracted["requested_date"] = None
        extracted["available_slots"] = None
        extracted["slot_id"] = None
        extracted["selected_slot_summary"] = None
        extracted["booking_summary"] = None
        extracted["confirmation_status"] = "not_requested"
    elif selected_slot is None:
        parsed_requested_date = parse_requested_date(entity_message)
        if state.get("date_clarification") is not None:
            extracted["date_clarification"] = None

    previous_date = state.get("requested_date")

    if parsed_requested_date is not None:
        if state.get("clarification_attempts"):
            extracted["clarification_attempts"] = 0
        if (
            previous_date is not None
            and parsed_requested_date != previous_date
        ):
            extracted["available_slots"] = None
            extracted["upcoming_alternative_slots"] = None
            extracted["slot_id"] = None
            extracted["selected_slot_summary"] = None
            extracted["booking_summary"] = None
            extracted["confirmation_status"] = "not_requested"

        extracted["requested_date"] = parsed_requested_date

    time_preference, time_preference_error = extract_time_preference(
        entity_message,
    )

    if time_preference_error is not None:
        extracted["time_preference"] = None
        extracted["time_preference_error"] = time_preference_error
    elif time_preference is not None:
        extracted["time_preference"] = time_preference
        extracted["time_preference_error"] = None

        # A new time preference must replace the previously selected
        # slot and force availability to be checked again.
        extracted["available_slots"] = None
        extracted["upcoming_alternative_slots"] = None
        extracted["slot_id"] = None
        extracted["selected_slot_summary"] = None
        extracted["booking_summary"] = None
        extracted["confirmation_status"] = "not_requested"
        extracted["slot_selection_error"] = None
        extracted["tool_error"] = None

    if (
        state.get("confirmation_status") == "pending"
        and state.get("intent") == "book_appointment"
        and selected_slot is None
        and is_time_correction_message(user_message)
    ):
        extracted["slot_id"] = None
        extracted["selected_slot_summary"] = None
        extracted["booking_summary"] = None
        extracted["confirmation_status"] = "not_requested"
        extracted["appointment_id"] = None
        extracted["appointment_reference_number"] = None

    if state.get("intent") in {
        "book_appointment",
        "cancel_appointment",
        "view_appointments",
    }:
        phone_candidate = extract_phone_candidate(user_message)

        if phone_candidate is not None:
            phone_number = normalize_sri_lankan_phone_number(
                phone_candidate,
            )

            if phone_number is None:
                extracted["customer_phone_invalid"] = True
                extracted["customer_phone_number"] = None
            else:
                extracted["customer_phone_invalid"] = False
                extracted["customer_phone_number"] = phone_number

                if (
                    state.get("intent") == "book_appointment"
                    and state.get("customer_phone_number") is not None
                    and phone_number
                    != state.get("customer_phone_number")
                ):
                    extracted["customer_id"] = None
                    extracted["booking_summary"] = None
                    extracted["confirmation_status"] = "not_requested"

            if state.get("intent") == "book_appointment":
                explicit_name_supplied = (
                    re.search(
                        r"\b(?:my\s+name\s+is|name\s+is|"
                        r"i\s+am|i['’]m|this\s+is)\b",
                        user_message,
                        flags=re.IGNORECASE,
                    )
                    is not None
                )

                phone_only_correction = (
                    is_phone_correction_message(user_message)
                    and not explicit_name_supplied
                )

                if not phone_only_correction:
                    customer_name = extract_customer_name(
                        user_message=user_message,
                        phone_number=phone_candidate,
                    )

                    if customer_name is not None:
                        extracted["customer_name"] = customer_name

                        if (
                            state.get("customer_name") is not None
                            and customer_name
                            != state.get("customer_name")
                        ):
                            extracted["customer_id"] = None
                            extracted["booking_summary"] = None
                            extracted["confirmation_status"] = (
                                "not_requested"
                            )

        elif (
            state.get("intent") == "book_appointment"
            and state.get("slot_id") is not None
            and re.search(
                r"\b(?:my\s+name\s+is|name\s+is|i\s+am|this\s+is)\b",
                user_message,
                flags=re.IGNORECASE,
            )
        ):
            customer_name = extract_customer_name(
                user_message=user_message,
                phone_number=None,
            )

            if customer_name is not None:
                extracted["customer_name"] = customer_name
                if (
                    state.get("customer_name") is not None
                    and customer_name != state.get("customer_name")
                ):
                    extracted["customer_id"] = None
                    extracted["booking_summary"] = None
                    extracted["confirmation_status"] = "not_requested"

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


def resolve_named_entities(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Resolve natural service and staff names into database IDs."""

    intent = state.get("intent")

    if intent not in {
        "book_appointment",
        "check_availability",
        "list_services",
        "reschedule_appointment",
    }:
        return {}

    user_message = get_latest_user_message(state)
    entity_message = entity_source_message(state, user_message)
    services = get_active_services()
    updates: AppointmentAgentState = {
        "available_services": services,
    }

    allow_numeric_service_choice = (
        state.get("service_id") is None
        and not state.get("available_slots")
    )

    matched_service = find_service_from_message(
        user_message=entity_message,
        services=services,
        allow_numeric_choice=allow_numeric_service_choice,
        prefer_last_match=is_targeted_service_correction_message(
            user_message
        ),
    )

    if matched_service is not None:
        if state.get("clarification_attempts"):
            updates["clarification_attempts"] = 0
        previous_service_id = state.get("service_id")
        new_service_id = int(matched_service["id"])

        updates["service_id"] = new_service_id
        updates["service_name"] = str(matched_service["name"])

        if previous_service_id != new_service_id:
            updates["available_slots"] = None
            updates["upcoming_alternative_slots"] = None
            updates["slot_id"] = None
            updates["selected_slot_summary"] = None
            updates["staff_id"] = None
            updates["staff_name"] = None
            updates["booking_summary"] = None
            updates["confirmation_status"] = "not_requested"

    elif (
        state.get("service_id") is not None
        and state.get("service_name") is None
    ):
        service = get_service_by_id(
            int(state["service_id"]),
        )

        if service is not None:
            updates["service_name"] = str(service["name"])

    resolved_service_id = updates.get(
        "service_id",
        state.get("service_id"),
    )

    if resolved_service_id is None:
        return updates

    staff_members = get_active_staff_for_service(
        int(resolved_service_id),
    )

    matched_staff = find_staff_from_message(
        user_message=entity_message,
        staff_members=staff_members,
    )

    if matched_staff is not None:
        previous_staff_id = state.get("staff_id")
        new_staff_id = int(matched_staff["id"])
        updates["staff_id"] = new_staff_id
        updates["staff_name"] = str(matched_staff["full_name"])

        if previous_staff_id != new_staff_id:
            updates["available_slots"] = None
            updates["upcoming_alternative_slots"] = None
            updates["slot_id"] = None
            updates["selected_slot_summary"] = None
            updates["booking_summary"] = None
            updates["confirmation_status"] = "not_requested"

    elif (
        state.get("staff_id") is None
        and updates.get("staff_id") is None
        and len(staff_members) == 1
    ):
        only_staff = staff_members[0]
        updates["staff_id"] = int(only_staff["id"])
        updates["staff_name"] = str(only_staff["full_name"])

    return updates

def resolve_customer_details(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Resolve customer details into a customer ID."""

    if state.get("intent") != "book_appointment":
        return {}

    if state.get("slot_id") is None:
        return {}

    if state.get("customer_id") is not None:
        return {}

    customer_name = state.get("customer_name")
    phone_number = state.get("customer_phone_number")

    if customer_name is None or phone_number is None:
        return {}

    normalized_phone_number = normalize_sri_lankan_phone_number(
        phone_number,
    )

    if normalized_phone_number is None:
        return {
            "customer_phone_number": None,
            "customer_phone_invalid": True,
        }

    customer = get_or_create_customer_from_details(
        full_name=customer_name,
        phone_number=normalized_phone_number,
    )

    return {
        "customer_id": int(customer["id"]),
        "customer_name": str(customer["full_name"]),
        "customer_phone_number": (
            normalize_sri_lankan_phone_number(
                str(customer["phone_number"]),
            )
            or normalized_phone_number
        ),
        "customer_phone_invalid": False,
    }

def confirm_or_reject_booking(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Execute a booking, cancellation, or reschedule after confirmation."""

    intent = state.get("intent")

    if intent == "cancel_appointment":
        if state.get("confirmation_status") == "rejected":
            return {
                "next_question": (
                    "No problem, I've kept your appointment as it is."
                ),
                "booking_summary": None,
            }

        if (
            state.get("confirmation_status") != "confirmed"
            or state.get("booking_summary") is None
        ):
            return {}

        if state.get("appointment_id") is None:
            return {
                "confirmation_status": "not_requested",
                "next_question": (
                    "Sure, I can help cancel an appointment. Please "
                    "share your phone number or appointment reference."
                ),
            }

        try:
            appointment = cancel_confirmed_appointment_from_state(state)

        except AppointmentNotFoundError:
            return {
                "confirmation_status": "not_requested",
                "next_question": (
                    "I couldn't find an appointment with that reference. "
                    "Please check the reference number or share your "
                    "phone number."
                ),
            }

        except InvalidAppointmentError as error:
            response = (
                "This appointment was already cancelled."
                if "confirmed" in str(error).lower()
                else f"I could not cancel this appointment: {error}"
            )

            return {
                "confirmation_status": "not_requested",
                "next_question": response,
            }

        return {
            "appointment_status": str(
                appointment.get("appointment_status") or "CANCELLED"
            ),
            "confirmation_status": "confirmed",
            "booking_summary": None,
            "next_question": (
                "Your appointment has been cancelled. Reference: "
                f"{appointment['appointment_reference_number']}."
            ),
        }

    if intent == "reschedule_appointment":
        if state.get("confirmation_status") == "rejected":
            return {
                "next_question": (
                    "No problem. Your original appointment stays as is."
                ),
                "booking_summary": None,
            }

        if (
            state.get("confirmation_status") != "confirmed"
            or state.get("booking_summary") is None
        ):
            return {}

        required_fields = ["appointment_id", "slot_id"]
        missing_fields = [
            field_name
            for field_name in required_fields
            if state.get(field_name) is None
        ]

        if missing_fields:
            return {
                "confirmation_status": "pending",
                "next_question": (
                    "I still need the new appointment slot before "
                    "I can confirm the reschedule."
                ),
            }

        if (
            state.get("current_slot_id") is not None
            and state.get("slot_id") == state.get("current_slot_id")
        ):
            return {
                "confirmation_status": "confirmed",
                "booking_summary": None,
                "next_question": (
                    "That's already your current appointment time, so "
                    "no changes are needed."
                ),
            }

        try:
            appointment = reschedule_confirmed_appointment_from_state(
                state,
            )

        except (AppointmentConflictError, InvalidAppointmentError) as error:
            return {
                "confirmation_status": "pending",
                "next_question": (
                    "I could not reschedule this appointment because "
                    f"{error} Please choose another available slot."
                ),
            }

        final_response = format_rescheduled_appointment_response(
            state=state,
            appointment=appointment,
        )

        return {
            "appointment_id": int(appointment["id"]),
            "appointment_reference_number": str(
                appointment["reference_number"]
            ),
            "current_slot_id": int(
                appointment.get("slot_id") or state["slot_id"]
            ),
            "confirmation_status": "confirmed",
            "next_question": final_response,
        }

    if intent != "book_appointment":
        return {}

    if state.get("appointment_id") is not None:
        return {}

    if state.get("confirmation_status") == "rejected":
        return {
            "next_question": (
                "No problem. I have not booked this appointment. "
                "You can choose a different service, date, or slot."
            ),
            "booking_summary": None,
        }

    if (
        state.get("confirmation_status") != "confirmed"
        or state.get("booking_summary") is None
    ):
        return {}

    required_fields = ["customer_id", "service_id", "staff_id", "slot_id"]
    missing_fields = [
        field_name
        for field_name in required_fields
        if state.get(field_name) is None
    ]

    if missing_fields:
        return {
            "confirmation_status": "pending",
            "next_question": (
                "I still need a little more information before I can "
                "confirm the booking."
            ),
        }

    try:
        appointment = create_confirmed_appointment_from_state(state)

    except (AppointmentConflictError, InvalidAppointmentError) as error:
        return {
            "confirmation_status": "pending",
            "next_question": (
                "I could not confirm this appointment because "
                f"{error} Please choose another available slot."
            ),
        }

    final_response = format_confirmed_appointment_response(
        state=state,
        appointment=appointment,
    )

    return {
        "appointment_id": int(appointment["id"]),
        "appointment_reference_number": str(
            appointment["reference_number"]
        ),
        "current_slot_id": int(
            appointment.get("slot_id") or state["slot_id"]
        ),
        "confirmation_status": "confirmed",
        "next_question": final_response,
    }


def lookup_conversation_availability(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Check real availability once service and date are available."""

    intent = state.get("intent")

    if intent not in {
        "book_appointment",
        "check_availability",
        "reschedule_appointment",
    }:
        return {}

    service_id = state.get("service_id")

    latest_user_message = get_latest_user_message(state)

    if is_upcoming_availability_request(latest_user_message):
        if service_id is None:
            return {}

        try:
            slots = find_upcoming_available_slots(
                service_id=int(service_id),
                staff_id=(
                    int(state["staff_id"])
                    if state.get("staff_id") is not None
                    else None
                ),
            )
        except Exception:
            return {
                "available_slots": None,
                "tool_error": (
                    "I couldn't verify availability right now. Please "
                    "try again or contact the front desk."
                ),
            }

        # When the user asks for "other" slots, do not repeat slots
        # that were already displayed in the previous response.
        if "other" in normalize_text(latest_user_message):
            shown_slot_ids = {
                int(slot["slot_id"])
                for slot in (state.get("available_slots") or [])
                if isinstance(slot, dict)
                and slot.get("slot_id") is not None
            }

            slots = [
                slot
                for slot in slots
                if int(slot["slot_id"]) not in shown_slot_ids
            ]

        return {
            "requested_date": None,
            "available_slots": slots,
            "upcoming_alternative_slots": None,
            "slot_id": None,
            "selected_slot_summary": None,
            "booking_summary": None,
            "confirmation_status": "not_requested",
            "tool_error": None,
        }

    requested_date = state.get("requested_date")
    slot_id = state.get("slot_id")

    if slot_id is not None:
        return {}

    if service_id is None or requested_date is None:
        return {}

    tool_arguments: dict[str, object] = {
        "service_id": service_id,
        "requested_date": requested_date,
    }

    staff_id = state.get("staff_id")

    if staff_id is not None:
        tool_arguments["staff_id"] = staff_id

    try:
        slots = get_validated_available_slots(tool_arguments)
    except Exception:
        return {
            "available_slots": None,
            "tool_error": (
                "I couldn't verify availability right now. Please try "
                "again or contact the front desk."
            ),
        }

    time_preference = state.get("time_preference")

    if time_preference is not None:
        matching_slots = filter_slots_by_time_preference(
            slots,
            time_preference,
        )

        if not matching_slots and slots:
            return {
                "available_slots": slots,
                "time_preference_error": (
                    "I found availability on that date, but none of the "
                    f"slots match {time_preference.get('label', 'that time')}. "
                    "Please choose another time preference or one of the "
                    "available slots."
                ),
            }

        slots = matching_slots

    if not slots and intent == "book_appointment":
        alternative_slots = find_upcoming_available_slots(
            service_id=int(service_id),
            staff_id=(
                int(staff_id)
                if staff_id is not None
                else None
            ),
        )

        return {
            "available_slots": alternative_slots,
            "upcoming_alternative_slots": alternative_slots,
        }

    selected_slot = (
        slots[0]
        if time_preference is not None and len(slots) == 1
        else find_slot_from_message(
            user_message=get_latest_user_message(state),
            available_slots=slots,
        )
        if time_preference is None
        else None
    )

    if selected_slot is not None:
        selected_start = parse_datetime(
            selected_slot.get("start_datetime"),
        )

        return {
            "available_slots": slots,
            "slot_id": int(selected_slot["slot_id"]),
            "staff_id": int(selected_slot["staff_id"]),
            "staff_name": (
                str(selected_slot["staff_name"])
                if selected_slot.get("staff_name") is not None
                else None
            ),
            "selected_slot_summary": format_slot_summary(selected_slot),
            "requested_date": (
                selected_start.date().isoformat()
                if selected_start is not None
                else requested_date
            ),
            "slot_selection_error": None,
            "time_preference_error": None,
            "tool_error": None,
        }

    return {
        "available_slots": slots,
        "tool_error": None,
    }


def calculate_missing_fields(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Calculate the information still required for the current intent."""

    intent = state.get("intent")
    missing_fields: list[str] = []

    if intent == "book_appointment":
        required_fields = (
            "service_id",
            "requested_date",
            "slot_id",
            "customer_id",
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


def determine_next_question(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Choose the next controlled response for the appointment flow."""

    intent = state.get("intent")
    next_question: str | None = None

    user_message = get_latest_user_message(state)

    if state.get("state_expired_message") is not None:
        return {
            "next_question": state["state_expired_message"],
        }

    if state.get("messages") and not user_message.strip():
        return {
            "next_question": (
                "Please type a message so I can help with your "
                "appointment."
            ),
        }

    nlu_result = nlu_result_for_response(state, user_message)
    response_intent = informational_response_intent(state)

    if nlu_result.intent == "ask_notification_capability":
        return {
            "intent": response_intent,
            "next_question": (
                "Appointment reminders and notifications are not enabled "
                "in this demo yet.\n"
                "Right now I can help you book, reschedule, cancel, check "
                "availability, and view appointments.\n"
                "Reminders can be added later using background jobs, SMS, "
                "email, or calls."
            ),
        }

    if nlu_result.intent == "ask_service_availability":
        services = state.get("available_services") or get_active_services()

        matched_service = find_service_from_message(
            user_message=user_message,
            services=services,
            allow_numeric_choice=False,
        )

        if matched_service is None:
            lower_message = user_message.lower()

            if any(word in lower_message for word in ["tooth", "teeth", "dentist", "dental"]):
                matched_service = next(
                    (
                        service
                        for service in services
                        if "dental" in str(service.get("name", "")).lower()
                    ),
                    None,
                )

            elif any(word in lower_message for word in ["skin", "dermatology"]):
                matched_service = next(
                    (
                        service
                        for service in services
                        if "dermatology" in str(service.get("name", "")).lower()
                    ),
                    None,
                )

            elif any(word in lower_message for word in ["physio", "physiotherapy"]):
                matched_service = next(
                    (
                        service
                        for service in services
                        if "physio" in str(service.get("name", "")).lower()
                    ),
                    None,
                )

        if matched_service is None:
            return {
                "intent": response_intent,
                "next_question": (
                    "I don't see that service listed.\n\n"
                    "These are the services currently available:\n\n"
                    + format_service_options(services)
                ),
            }

        price = matched_service.get("price")
        price_text = (
            f"LKR {float(price):,.0f}"
            if price is not None
            else "not currently listed"
        )

        return {
            "intent": response_intent,
            "next_question": (
                f"Yes, we offer {matched_service['name']}.\n\n"
                f"{matched_service['name']} — "
                f"{matched_service.get('description') or 'Service details are available.'}\n"
                f"Duration: {matched_service.get('duration_minutes')} minutes\n"
                f"Price: {price_text}\n\n"
                "Would you like to book it or check availability?"
            ),
        }
    if nlu_result.intent == "ask_pricing":
        services = state.get("available_services") or get_active_services()

        matched_service = find_service_from_message(
            user_message=user_message,
            services=services,
            allow_numeric_choice=False,
        )

        if matched_service is None:
            lower_message = user_message.lower()

            if any(
                word in lower_message
                for word in ["tooth", "teeth", "dentist", "dental"]
            ):
                matched_service = next(
                    (
                        service
                        for service in services
                        if "dental"
                        in str(service.get("name", "")).lower()
                    ),
                    None,
                )

            elif any(
                word in lower_message
                for word in ["skin", "dermatology"]
            ):
                matched_service = next(
                    (
                        service
                        for service in services
                        if "dermatology"
                        in str(service.get("name", "")).lower()
                    ),
                    None,
                )

            elif any(
                word in lower_message
                for word in ["physio", "physiotherapy"]
            ):
                matched_service = next(
                    (
                        service
                        for service in services
                        if "physio"
                        in str(service.get("name", "")).lower()
                    ),
                    None,
                )

        # For contextual questions such as "how much is it?",
        # reuse the verified active service.
        if (
            matched_service is None
            and state.get("service_id") is not None
        ):
            active_service_id = int(state["service_id"])

            matched_service = next(
                (
                    service
                    for service in services
                    if int(service.get("id", -1))
                    == active_service_id
                ),
                None,
            )

        if matched_service is None:
            return {
                "intent": response_intent,
                "next_question": (
                    "Which service price would you like to know?\n\n"
                    + format_service_options(services)
                ),
            }

        price = matched_service.get("price")
        price_text = (
            f"LKR {float(price):,.0f}"
            if price is not None
            else "not currently listed"
        )

        return {
            "intent": response_intent,
            "next_question": compose_informational_response(
                f"{matched_service['name']} costs {price_text}.",
                state,
            ),
        }

    if nlu_result.intent == "ask_duration":
        services = state.get("available_services") or get_active_services()

        matched_service = find_service_from_message(
            user_message=user_message,
            services=services,
            allow_numeric_choice=False,
        )

        if matched_service is None:
            lower_message = user_message.lower()

            if any(
                word in lower_message
                for word in ["tooth", "teeth", "dentist", "dental"]
            ):
                matched_service = next(
                    (
                        service
                        for service in services
                        if "dental"
                        in str(service.get("name", "")).lower()
                    ),
                    None,
                )

            elif any(
                word in lower_message
                for word in ["skin", "dermatology"]
            ):
                matched_service = next(
                    (
                        service
                        for service in services
                        if "dermatology"
                        in str(service.get("name", "")).lower()
                    ),
                    None,
                )

            elif any(
                word in lower_message
                for word in ["physio", "physiotherapy"]
            ):
                matched_service = next(
                    (
                        service
                        for service in services
                        if "physio"
                        in str(service.get("name", "")).lower()
                    ),
                    None,
                )

        # For contextual questions such as "how long does it take?",
        # reuse the verified active service.
        if (
            matched_service is None
            and state.get("service_id") is not None
        ):
            active_service_id = int(state["service_id"])

            matched_service = next(
                (
                    service
                    for service in services
                    if int(service.get("id", -1))
                    == active_service_id
                ),
                None,
            )

        if matched_service is None:
            return {
                "intent": response_intent,
                "next_question": (
                    "Which service duration would you like to know?\n\n"
                    + format_service_options(services)
                ),
            }

        return {
            "intent": response_intent,
            "next_question": compose_informational_response(
                f"{matched_service['name']} takes "
                f"{matched_service.get('duration_minutes')} minutes.",
                state,
            ),
        }

    if nlu_result.intent == "ask_service_list":
        services = state.get("available_services") or get_active_services()

        return {
            "intent": response_intent,
            "next_question": (
                "These are the services currently available:\n\n"
                + format_service_options(services)
                + "\n\nWould you like to book one of these or check availability?"
            ),
        }
    if nlu_result.intent in {"ask_opening_hours", "ask_location"}:
        return {
            "intent": response_intent,
            "next_question": compose_informational_response(
                "I don't have that information available yet. Please "
                "contact the front desk or clinic staff for accurate "
                "details.",
                state,
            ),
        }
    if nlu_result.intent in {
        "ask_insurance",
        "ask_cancellation_policy",
        "ask_payment_methods",
    }:
        return {
            "intent": response_intent,
            "next_question": compose_informational_response(
                "I don't have that information available yet. Please "
                "contact the front desk or clinic staff for accurate "
                "details.",
                state,
            ),
        }

    if is_human_handoff_message(user_message):
        return {
            "next_question": (
                "I can hand this over to a human staff member. "
                "Please contact the front desk or clinic staff to "
                "continue this request."
            ),
        }

    if is_graceful_exit_message(user_message):
        return {
            "next_question": format_graceful_exit_response(
                user_message,
            ),
        }

    if state.get("customer_phone_invalid"):
        return {
            "next_question": (
                "Please enter a valid Sri Lankan phone number, for "
                "example 0771234567 or +94771234567."
            ),
        }

    if state.get("time_preference_error") is not None:
        return {
            "next_question": state["time_preference_error"],
        }

    if state.get("date_clarification") is not None:
        return clarification_response(
            state,
            str(state["date_clarification"]),
        )

    if state.get("tool_error") is not None:
        return {
            "next_question": state["tool_error"],
        }

    if state.get("slot_selection_error") is not None:
        slot_options = state.get("available_slots") or []
        message = str(state["slot_selection_error"])
        if slot_options:
            message += "\n\n" + format_available_slots(slot_options)
        return clarification_response(state, message)

    if (
        state.get("confirmation_status") == "pending"
        and state.get("booking_summary") is not None
        and is_ambiguous_confirmation_message(user_message)
    ):
        return {
            "next_question": (
                "Please reply yes to confirm, or no to cancel/change "
                "this request."
            ),
        }

    if is_capabilities_request(user_message):
        return {
            "next_question": (
                "I can help you book, reschedule, or cancel an "
                "appointment, check availability, list services and "
                "prices, or view your appointments."
            ),
        }

    if is_greeting_or_small_talk(user_message):
        return {
            "next_question": (
                "I'm here and ready to help with appointments. You can "
                "ask me to book, reschedule, cancel, check availability, "
                "or view your appointments."
            ),
        }

    if is_safe_information_request(user_message):
        return {
            "next_question": (
                "I don't have that information available yet. Please "
                "contact the front desk or clinic staff for accurate "
                "details."
            ),
        }

    if is_pricing_request(user_message):
        services = state.get("available_services") or get_active_services()
        matched_service = find_service_from_message(
            user_message=user_message,
            services=services,
            allow_numeric_choice=False,
        )

        if matched_service is None:
            return {
                "next_question": (
                    "Which service price would you like to know?\n\n"
                    + format_service_options(services)
                ),
            }

        price = matched_service.get("price")
        price_text = (
            f"LKR {float(price):,.0f}"
            if price is not None
            else "not currently listed"
        )

        return {
            "next_question": compose_informational_response(
                f"{matched_service['name']} costs {price_text}.",
                state,
            ),
        }

    if is_upcoming_availability_request(user_message):
        service_id = state.get("service_id")

        if service_id is None:
            services = (
                state.get("available_services")
                or get_active_services()
            )

            return {
                "next_question": (
                    "Which service would you like to check?\n\n"
                    "Available services:\n"
                    + format_service_options(services)
                ),
            }

        service_name = state.get("service_name") or "appointment"
        available_slots = state.get("available_slots") or []

        return {
            "next_question": format_upcoming_available_slots(
                service_name=service_name,
                slots=available_slots,
            ),
        }

    if (
        state.get("appointment_id") is not None
        and is_thank_you_message(get_latest_user_message(state))
    ):
        return {
            "next_question": (
                "You're welcome. Let me know if you need help "
                "with anything else."
            ),
        }

    services = state.get("available_services") or get_active_services()

    if (
        intent == "book_appointment"
        and state.get("appointment_id") is not None
    ):
        return {
            "next_question": state.get("next_question"),
        }

    if (
        intent == "reschedule_appointment"
        and state.get("confirmation_status") in {
            "confirmed",
            "rejected",
        }
    ):
        return {
            "next_question": state.get("next_question"),
        }

    if (
        intent == "cancel_appointment"
        and state.get("confirmation_status") in {
            "confirmed",
            "rejected",
        }
    ):
        return {
            "next_question": state.get("next_question"),
        }

    if (
        intent == "book_appointment"
        and state.get("confirmation_status") == "rejected"
    ):
        return {
            "next_question": state.get("next_question"),
        }

    if intent == "view_appointments":
        status_resolved = False
        reference_number = extract_appointment_reference(user_message)

        if reference_number is not None:
            appointment = get_appointment_by_reference(reference_number)

            if appointment is None:
                next_question = (
                    "I couldn't find an appointment with that reference. "
                    "Please check the reference number or share your "
                    "phone number."
                )
            else:
                next_question = (
                    "Here is your appointment status:\n\n"
                    + format_appointment_details(appointment)
                )
                status_resolved = True

        elif (
            state.get("customer_id") is not None
            or state.get("customer_phone_number") is not None
        ):
            appointments = get_upcoming_appointments_for_customer(
                customer_id=(
                    int(state["customer_id"])
                    if state.get("customer_id") is not None
                    else None
                ),
                phone_number=state.get("customer_phone_number"),
            )
            next_question = format_appointment_list(appointments)
            status_resolved = True

        else:
            next_question = (
                "Sure, I can check that. Please share your phone number "
                "or appointment reference."
            )

        paused_intent = state.get("paused_intent")
        if status_resolved and paused_intent in ACTIVE_FLOW_INTENTS:
            resumed_state: AppointmentAgentState = dict(state)
            resumed_state["intent"] = paused_intent
            resume_prompt = format_resume_prompt(resumed_state)
            if resume_prompt is not None:
                next_question = f"{next_question}\n\n{resume_prompt}"
            return {
                "intent": paused_intent,
                "paused_intent": None,
                "next_question": next_question,
            }

    elif intent == "list_services":
        next_question = (
            "These are the services currently available:\n\n"
            + format_service_options(services)
            + "\n\nWould you like to book one of these?"
        )

    elif intent == "book_appointment":
        if state.get("service_id") is None:
            next_question = (
                "Sure, I can help you book an appointment.\n\n"
                "Which service would you like?\n\n"
                "Available services:\n"
                + format_service_options(services)
            )

        elif state.get("requested_date") is None:
            service_name = state.get("service_name") or "that service"
            staff_members = get_active_staff_for_service(
                int(state["service_id"]),
            )

            if len(staff_members) == 1:
                staff_note = (
                    f"{service_name} is currently handled by "
                    f"{staff_members[0]['full_name']}."
                )
            elif staff_members:
                staff_lines = [
                    (
                        f"{index}. {staff['full_name']} — "
                        f"{staff.get('speciality') or 'Staff member'}"
                    )
                    for index, staff in enumerate(
                        staff_members,
                        start=1,
                    )
                ]

                staff_note = (
                    "Available doctors/staff for this service:\n"
                    + "\n".join(staff_lines)
                    + "\n\nYou can mention a preferred doctor/staff "
                    "member, or I can show all available slots."
                )
            else:
                staff_note = (
                    "I will check the available staff for this service."
                )

            next_question = (
                f"Sure, I can help you book {service_name}.\n\n"
                f"{staff_note}\n\n"
                "Which date would you prefer? "
                "You can say tomorrow, next Monday, or use YYYY-MM-DD."
            )

        elif state.get("slot_id") is None:
            available_slots = state.get("available_slots")
            alternative_slots = state.get("upcoming_alternative_slots")
            service_name = state.get("service_name") or "appointment"
            friendly_date = format_date(
                state.get("requested_date"),
            )

            if alternative_slots:
                next_question = (
                    f"There are no available {service_name} slots "
                    f"for {friendly_date}.\n\n"
                    + format_upcoming_available_slots(
                        service_name=service_name,
                        slots=alternative_slots,
                    )
                    + "\n\nWhich one would you prefer?"
                )
            elif not available_slots:
                next_question = (
                    f"There are no available {service_name} slots "
                    f"for {friendly_date}.\n\n"
                    "Which other date would you prefer?"
                )
            else:
                next_question = (
                    f"I found these {service_name} slots for "
                    f"{friendly_date}:\n\n"
                    + format_available_slots(available_slots)
                    + "\n\nWhich one would you prefer?"
                )

        elif state.get("customer_id") is None:
            selected_slot_summary = (
                state.get("selected_slot_summary")
                or "the selected appointment slot"
            )

            if (
                state.get("customer_name") is not None
                and state.get("customer_phone_number") is None
            ):
                next_question = (
                    f"Great. You selected {selected_slot_summary}.\n\n"
                    "Please enter a valid Sri Lankan phone number, for "
                    "example 0771234567 or +94771234567."
                )
            elif (
                state.get("customer_name") is None
                and state.get("customer_phone_number") is not None
            ):
                next_question = (
                    f"Great. You selected {selected_slot_summary}.\n\n"
                    "May I have your full name for the booking?"
                )
            else:
                next_question = (
                    f"Great. You selected {selected_slot_summary}.\n\n"
                    "May I have your full name and phone number "
                    "for the booking?"
                )

        elif state.get("confirmation_status") != "pending":
            booking_summary = format_booking_confirmation_summary(
                state,
            )

            next_question = booking_summary

            return {
                "next_question": next_question,
                "booking_summary": booking_summary,
                "confirmation_status": "pending",
            }

    elif intent == "check_availability":
        if state.get("service_id") is None:
            next_question = (
                "Which service would you like to check?\n\n"
                "Available services:\n"
                + format_service_options(services)
            )

        elif state.get("requested_date") is None:
            service_name = state.get("service_name") or "that service"

            next_question = (
                f"Which date would you like to check for "
                f"{service_name}?"
            )

        else:
            available_slots = state.get("available_slots")
            service_name = state.get("service_name") or "appointment"
            friendly_date = format_date(
                state.get("requested_date"),
            )

            if not available_slots:
                next_question = (
                    f"There are no available {service_name} slots "
                    f"for {friendly_date}."
                )
            else:
                next_question = (
                    f"These {service_name} slots are available "
                    f"for {friendly_date}:\n\n"
                    + format_available_slots(available_slots)
                )

    elif intent == "cancel_appointment":
        appointment: dict[str, object] | None = None

        if state.get("appointment_id") is not None:
            appointment = get_appointment_by_id_for_conversation(
                int(state["appointment_id"]),
            )

        elif state.get("appointment_reference_number") is not None:
            appointment = get_appointment_by_reference(
                str(state["appointment_reference_number"]),
            )

        if appointment is None and (
            state.get("customer_id") is not None
            or state.get("customer_phone_number") is not None
        ):
            appointments = get_upcoming_appointments_for_customer(
                customer_id=(
                    int(state["customer_id"])
                    if state.get("customer_id") is not None
                    else None
                ),
                phone_number=state.get("customer_phone_number"),
            )

            if len(appointments) == 1:
                appointment = appointments[0]
            elif len(appointments) > 1:
                next_question = (
                    format_appointment_list(appointments)
                    + "\n\nWhich appointment reference would you like "
                    "to cancel?"
                )
            else:
                next_question = (
                    "I couldn't find any upcoming appointments for "
                    "those details."
                )

        if appointment is not None:
            appointment_status = str(
                appointment.get("appointment_status") or ""
            )

            if appointment_status != AppointmentStatus.CONFIRMED.value:
                next_question = "This appointment was already cancelled."
            elif state.get("confirmation_status") != "pending":
                booking_summary = (
                    format_cancellation_confirmation_summary(appointment)
                )

                return {
                    "appointment_id": int(
                        appointment["appointment_id"]
                    ),
                    "appointment_reference_number": str(
                        appointment["appointment_reference_number"]
                    ),
                    "appointment_status": appointment_status,
                    "current_slot_id": int(
                        appointment["current_slot_id"]
                    ),
                    "service_id": int(appointment["service_id"]),
                    "service_name": str(appointment["service_name"]),
                    "staff_id": int(appointment["staff_id"]),
                    "staff_name": str(appointment["staff_name"]),
                    "next_question": booking_summary,
                    "booking_summary": booking_summary,
                    "confirmation_status": "pending",
                }

        elif next_question is None:
            if state.get("appointment_reference_number") is not None:
                next_question = (
                    "I couldn't find an appointment with that reference. "
                    "Please check the reference number or share your "
                    "phone number."
                )
            else:
                next_question = (
                    "Sure, I can help cancel an appointment. Please "
                    "share your phone number or appointment reference."
                )

    elif intent == "reschedule_appointment":
        service_name = state.get("service_name") or "appointment"
        appointment_label = (
            f"{service_name} appointment"
            if service_name != "appointment"
            else "this appointment"
        )
        friendly_date = format_date(
            state.get("requested_date"),
        )

        if state.get("appointment_id") is None:
            next_question = (
                "Sure, I can help reschedule an appointment. "
                "What is your appointment ID or reference number?"
            )

        elif state.get("requested_date") is None:
            next_question = (
                f"Which date would you like to move "
                f"{appointment_label} to?"
            )

        elif state.get("slot_id") is None:
            available_slots = state.get("available_slots")

            if not available_slots:
                next_question = (
                    f"There are no available {service_name} slots "
                    f"for {friendly_date}.\n\n"
                    "Which other date would you prefer?"
                )
            else:
                next_question = (
                    f"I found these available {service_name} slots "
                    f"for {friendly_date}:\n\n"
                    + format_available_slots(available_slots)
                    + "\n\nWhich new slot would you prefer?"
                )

        elif state.get("confirmation_status") != "pending":
            booking_summary = format_reschedule_confirmation_summary(
                state,
            )

            next_question = booking_summary

            return {
                "next_question": next_question,
                "booking_summary": booking_summary,
                "confirmation_status": "pending",
            }

    elif intent == "general_question":
        if is_likely_gibberish(user_message):
            next_question = (
                "I didn't quite catch that. I can help you book, "
                "reschedule, cancel, check availability, or view "
                "appointments."
            )

    return {
        "next_question": next_question,
    }


def call_model(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Return a controlled question or generate a model response."""

    next_question = state.get("next_question")

    if next_question is not None:
        return {
            "messages": [
                LangChainAIMessage(content=next_question),
            ],
        }

    try:
        model = get_chat_model()

        if hasattr(model, "bind_tools"):
            model = model.bind_tools(READ_ONLY_TOOLS)

        response = model.invoke(
            [
                SystemMessage(content=APPOINTMENT_SYSTEM_PROMPT),
                *state["messages"],
            ]
        )

    except Exception:
        response = LangChainAIMessage(
            content=(
                "I'm having trouble reaching the AI model right now, "
                "but I can still help with appointments. You can ask "
                "me to book, reschedule, cancel, check availability, "
                "or view your appointments."
            )
        )

    return {
        "messages": [response],
    }


def build_appointment_agent(
    checkpointer: InMemorySaver | None = None,
):
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
        "resolve_named_entities",
        resolve_named_entities,
    )

    graph_builder.add_node(
        "resolve_customer_details",
        resolve_customer_details,
    )
    graph_builder.add_node(
        "confirm_or_reject_booking",
        confirm_or_reject_booking,
    )

    graph_builder.add_node(
        "calculate_missing_fields",
        calculate_missing_fields,
    )

    graph_builder.add_node(
        "lookup_conversation_availability",
        lookup_conversation_availability,
    )
    graph_builder.add_node(
        "resolve_customer_after_availability",
        resolve_customer_details,
    )

    graph_builder.add_node(
        "determine_next_question",
        determine_next_question,
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
        "resolve_named_entities",
    )

    graph_builder.add_edge(
        "resolve_named_entities",
        "resolve_customer_details",
    )

    graph_builder.add_edge(
        "resolve_customer_details",
        "confirm_or_reject_booking",
    )

    graph_builder.add_edge(
        "confirm_or_reject_booking",
        "calculate_missing_fields",
    )

    graph_builder.add_edge(
        "calculate_missing_fields",
        "lookup_conversation_availability",
    )

    graph_builder.add_edge(
        "lookup_conversation_availability",
        "resolve_customer_after_availability",
    )

    graph_builder.add_edge(
        "resolve_customer_after_availability",
        "determine_next_question",
    )

    graph_builder.add_edge(
        "determine_next_question",
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

    if checkpointer is None:
        return graph_builder.compile()

    return graph_builder.compile(checkpointer=checkpointer)


appointment_agent = build_appointment_agent(InMemorySaver())
persistent_appointment_agent = build_appointment_agent()


def serialize_agent_state(
    state: AppointmentAgentState,
) -> dict:
    """Create a JSON-safe checkpoint without duplicating message rows."""

    return {
        key: value
        for key, value in state.items()
        if key != "messages"
        and isinstance(value, (str, int, float, bool, list, dict, type(None)))
    }


def restore_conversation_messages(
    database,
    conversation_id: int,
) -> list[AnyMessage]:
    """Restore persisted user/assistant history for a fresh agent process."""

    restored_messages: list[AnyMessage] = []

    for message in list_conversation_messages(database, conversation_id):
        if message.role == AIMessageRole.USER:
            restored_messages.append(HumanMessage(content=message.content))
        elif message.role == AIMessageRole.SYSTEM:
            restored_messages.append(SystemMessage(content=message.content))
        else:
            restored_messages.append(
                LangChainAIMessage(content=message.content)
            )

    return restored_messages


def parse_state_timestamp(value: object) -> datetime | None:
    """Parse a persisted state timestamp as an aware UTC datetime."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def timestamp_is_expired(
    value: object,
    ttl_minutes: int,
    now: datetime,
) -> bool:
    """Return True when a valid timestamp is older than its TTL."""

    timestamp = parse_state_timestamp(value)
    if timestamp is None:
        return False

    return now - timestamp > timedelta(minutes=ttl_minutes)


def expire_stale_transaction_state(
    state: AppointmentAgentState,
    now: datetime | None = None,
) -> AppointmentAgentState:
    """Invalidate stale transactional choices before processing a new turn."""

    recovered: AppointmentAgentState = dict(state)
    recovered["state_expired_message"] = None

    if recovered.get("confirmation_status") in {"confirmed", "rejected"}:
        return recovered

    settings = get_settings()
    current_time = now or utc_now()
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    pending_confirmation_expired = (
        recovered.get("confirmation_status") == "pending"
        and recovered.get("booking_summary") is not None
        and timestamp_is_expired(
            recovered.get("pending_action_started_at")
            or recovered.get("transaction_updated_at"),
            settings.pending_confirmation_ttl_minutes,
            current_time,
        )
    )
    slot_options_expired = (
        recovered.get("available_slots") is not None
        and timestamp_is_expired(
            recovered.get("slot_options_updated_at")
            or recovered.get("transaction_updated_at"),
            settings.availability_options_ttl_minutes,
            current_time,
        )
    )

    if pending_confirmation_expired or slot_options_expired:
        recovered.update(
            {
                "available_slots": None,
                "upcoming_alternative_slots": None,
                "slot_id": None,
                "staff_id": None,
                "staff_name": None,
                "selected_slot_summary": None,
                "booking_summary": None,
                "confirmation_status": "not_requested",
                "pending_action_started_at": None,
                "slot_options_updated_at": None,
                "state_expired_message": (
                    "That slot selection has expired, so I checked "
                    "availability again. Please choose one of the current "
                    "available options before confirming."
                ),
            }
        )
        return recovered

    if (
        recovered.get("intent") in ACTIVE_FLOW_INTENTS
        and timestamp_is_expired(
            recovered.get("transaction_updated_at"),
            settings.active_transaction_ttl_minutes,
            current_time,
        )
    ):
        recovered.update(
            {
                "intent": "general_question",
                "service_id": None,
                "service_name": None,
                "staff_id": None,
                "staff_name": None,
                "requested_date": None,
                "time_preference": None,
                "time_preference_error": None,
                "available_slots": None,
                "upcoming_alternative_slots": None,
                "slot_id": None,
                "selected_slot_summary": None,
                "booking_summary": None,
                "appointment_id": None,
                "appointment_reference_number": None,
                "appointment_status": None,
                "current_slot_id": None,
                "cancellation_reason": None,
                "slot_selection_error": None,
                "tool_error": None,
                "confirmation_status": "not_requested",
                "transaction_updated_at": None,
                "pending_action_started_at": None,
                "slot_options_updated_at": None,
                "state_expired_message": (
                    "That appointment request expired for safety. Please "
                    "start the booking, cancellation, rescheduling, or "
                    "availability request again."
                ),
            }
        )

    return recovered


def update_transaction_timestamps(
    result: AppointmentAgentState,
    previous_state: AppointmentAgentState,
    now: datetime | None = None,
) -> None:
    """Attach durable timestamps to active choices and pending actions."""

    timestamp = (now or utc_now()).astimezone(timezone.utc).isoformat()
    intent = result.get("intent")
    completed = result.get("confirmation_status") in {
        "confirmed",
        "rejected",
    }

    if intent in ACTIVE_FLOW_INTENTS and not completed:
        result["transaction_updated_at"] = timestamp
    else:
        result["transaction_updated_at"] = None

    if (
        result.get("confirmation_status") == "pending"
        and result.get("booking_summary") is not None
    ):
        result["pending_action_started_at"] = (
            previous_state.get("pending_action_started_at") or timestamp
        )
    else:
        result["pending_action_started_at"] = None

    if result.get("available_slots") is not None:
        if (
            result.get("available_slots")
            != previous_state.get("available_slots")
            or previous_state.get("slot_options_updated_at") is None
        ):
            result["slot_options_updated_at"] = timestamp
        else:
            result["slot_options_updated_at"] = previous_state.get(
                "slot_options_updated_at"
            )
    else:
        result["slot_options_updated_at"] = None


def get_conversation_stage(state: dict) -> str:
    """Derive a stable client-facing stage from durable state."""

    intent = state.get("intent")

    if state.get("confirmation_status") == "pending":
        return "awaiting_confirmation"

    if intent == "book_appointment":
        if state.get("service_id") is None:
            return "selecting_service"
        if state.get("requested_date") is None:
            return "selecting_date"
        if state.get("slot_id") is None:
            return "selecting_slot"
        if state.get("customer_id") is None:
            return "collecting_customer"
        return "ready_to_confirm"

    if intent == "reschedule_appointment":
        if state.get("appointment_id") is None:
            return "identifying_appointment"
        if state.get("requested_date") is None:
            return "selecting_date"
        if state.get("slot_id") is None:
            return "selecting_slot"
        return "ready_to_confirm"

    if intent == "cancel_appointment":
        return (
            "identifying_appointment"
            if state.get("appointment_id") is None
            else "ready_to_confirm"
        )

    if intent == "check_availability":
        return "checking_availability"

    if intent == "view_appointments":
        return "viewing_appointments"

    return "idle"


def get_structured_conversation_state(thread_id: str) -> dict:
    """Return safe structured metadata for the chat API response."""

    database = SessionLocal()

    try:
        conversation = get_or_create_conversation(database, thread_id)
        state = load_conversation_state(conversation)
        slots = state.get("available_slots")
        options = []

        if isinstance(slots, list):
            for slot in slots:
                if not isinstance(slot, dict) or slot.get("slot_id") is None:
                    continue
                options.append(
                    {
                        "id": int(slot["slot_id"]),
                        "label": (
                            f"{format_time(slot.get('start_datetime'))} with "
                            f"{slot.get('staff_name') or 'available staff'}"
                        ),
                        "start_datetime": slot.get("start_datetime"),
                        "end_datetime": slot.get("end_datetime"),
                    }
                )

        requires_confirmation = (
            state.get("confirmation_status") == "pending"
            and state.get("booking_summary") is not None
        )
        error = (
            state.get("tool_error")
            or state.get("time_preference_error")
            or state.get("slot_selection_error")
            or state.get("state_expired_message")
        )

        return {
            "intent": state.get("intent"),
            "conversation_stage": get_conversation_stage(state),
            "requires_confirmation": requires_confirmation,
            "pending_action": (
                state.get("intent") if requires_confirmation else None
            ),
            "options": options,
            "error": error,
        }
    finally:
        database.close()


def run_appointment_agent(
    user_message: str,
    thread_id: str | None = None,
    request_id: str | None = None,
) -> str:
    """Run and persist one turn of a stateful conversation."""

    resolved_thread_id = thread_id or str(uuid4())
    database = SessionLocal()
    request_execution = None

    try:
        conversation = get_or_create_conversation(
            database=database,
            thread_id=resolved_thread_id,
        )

        if request_id is not None:
            request_execution, cached_response = begin_request_execution(
                database,
                conversation.id,
                request_id,
            )

            if cached_response is not None:
                return cached_response

        restored_state: AppointmentAgentState = load_conversation_state(
            conversation
        )
        if (
            restored_state.get("transaction_updated_at") is None
            and restored_state.get("intent") in ACTIVE_FLOW_INTENTS
            and conversation.state_updated_at is not None
        ):
            state_updated_at = conversation.state_updated_at
            if state_updated_at.tzinfo is None:
                state_updated_at = state_updated_at.replace(
                    tzinfo=timezone.utc
                )
            restored_state["transaction_updated_at"] = (
                state_updated_at.astimezone(timezone.utc).isoformat()
            )

        restored_state = expire_stale_transaction_state(restored_state)
        restored_messages = restore_conversation_messages(
            database,
            conversation.id,
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

        result = persistent_appointment_agent.invoke(
            {
                **restored_state,
                "messages": [
                    *restored_messages,
                    HumanMessage(content=user_message),
                ]
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
        update_transaction_timestamps(result, restored_state)

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

        save_conversation_state(
            database=database,
            conversation=conversation,
            state_data=serialize_agent_state(result),
            current_intent=result.get("intent"),
            customer_id=result.get("customer_id"),
        )

        if request_execution is not None:
            complete_request_execution(
                database,
                request_execution,
                assistant_response,
            )

        return assistant_response

    except Exception as error:
        if request_execution is not None:
            fail_request_execution(database, request_execution, error)
        raise

    finally:
        database.close()
