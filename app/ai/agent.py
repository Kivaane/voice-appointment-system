import re
from datetime import date, datetime, timedelta
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
    get_or_create_conversation,
    record_event,
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
from app.ai.nlu import classify_message

READ_ONLY_TOOLS = [
    list_available_services,
    check_available_slots,
]


SYSTEM_PROMPT = """
You are a respectful AI appointment assistant.

Your responsibilities are to:
- help customers book appointments conversationally
- answer questions about available services
- check real appointment availability using tools
- ask for missing information clearly
- remember relevant details from earlier messages in the same thread
- respond like a helpful receptionist, not like a database form
- never ask customers for internal IDs unless there is no better option
- never invent appointment availability
- never claim that an appointment is booked unless a booking tool confirms it

Booking, cancellation, and rescheduling execution will be handled only
through confirmed tools.
""".strip()


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

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower(),
    ).strip()


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
                Appointment.start_datetime >= datetime.now(),
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
) -> dict[str, object] | None:
    """Resolve a service from natural text or numeric choice."""

    normalized_message = normalize_text(user_message)

    if allow_numeric_choice:
        number_match = re.fullmatch(
            r"\s*(\d+)\s*\.?\s*",
            user_message,
        )

        if number_match is not None:
            number = int(number_match.group(1))

            if 1 <= number <= len(services):
                return services[number - 1]

            for service in services:
                if service["id"] == number:
                    return service

    for service in services:
        service_name = str(service["name"])
        normalized_name = normalize_text(service_name)

        if normalized_name in normalized_message:
            return service

        service_words = [
            word
            for word in normalized_name.split()
            if len(word) >= 4
        ]

        if any(word in normalized_message for word in service_words):
            return service

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
    today = date.today()

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

            if "next" in normalized_message or days_ahead == 0:
                days_ahead += 7

            return (today + timedelta(days=days_ahead)).isoformat()

    return None

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

    hour = parsed_value.strftime("%I").lstrip("0")
    minute = parsed_value.strftime("%M")
    meridiem = parsed_value.strftime("%p")

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

    return any(
        phrase in normalized_message
        for phrase in date_change_phrases
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

    return any(
        phrase in normalized_message
        for phrase in service_change_phrases
    )


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
        "talk to a person",
        "speak to human",
        "front desk",
        "receptionist",
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


def is_upcoming_availability_request(
    user_message: str,
) -> bool:
    """Return True when the user asks for the next available dates."""

    normalized_message = normalize_text(user_message)
    upcoming_availability_phrases = (
        "which date available",
        "which dates are available",
        "tell me available dates",
        "show available dates",
        "any available date",
        "when are you available",
    )

    return any(
        phrase in normalized_message
        for phrase in upcoming_availability_phrases
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
    today = date.today()

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

        slots = check_available_slots.run(tool_arguments)
        upcoming_slots.extend(slots)

    return sorted(
        upcoming_slots,
        key=lambda slot: str(slot.get("start_datetime") or ""),
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

    user_message = get_latest_user_message(state).lower()

    nlu_result = classify_message(user_message)

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
        }

    if is_human_handoff_message(user_message):
        return {
            "intent": "general_question",
            "next_question": (
                "I can hand this over to a human staff member. "
                "Please contact the front desk or clinic staff to "
                "continue this request."
            ),
            "booking_summary": None,
            "available_slots": None,
            "slot_id": None,
            "selected_slot_summary": None,
            "confirmation_status": "not_requested",
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
        }

    if any(
        keyword in user_message
        for keyword in (
            "cancel",
            "remove my appointment",
        )
    ):
        return {
            "intent": "cancel_appointment",
            "booking_summary": None,
            "confirmation_status": "not_requested",
            "next_question": None,
        }

    if state.get("intent") == "reschedule_appointment":
        if is_explicit_new_booking_message(user_message):
            return {
                "intent": "book_appointment",
                "service_id": None,
                "service_name": None,
                "staff_id": None,
                "staff_name": None,
                "requested_date": None,
                "available_slots": None,
                "selected_slot_summary": None,
                "booking_summary": None,
                "slot_id": None,
                "appointment_id": None,
                "appointment_reference_number": None,
                "missing_fields": [],
                "next_question": None,
                "confirmation_status": "not_requested",
            }

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
        return {
            "intent": "book_appointment",
            "service_id": None,
            "service_name": None,
            "staff_id": None,
            "staff_name": None,
            "requested_date": None,
            "available_slots": None,
            "selected_slot_summary": None,
            "booking_summary": None,
            "slot_id": None,
            "appointment_id": None,
            "appointment_reference_number": None,
            "missing_fields": [],
            "next_question": None,
            "confirmation_status": "not_requested",
        }

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

        if is_service_correction_message(user_message):
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

        if is_service_correction_message(user_message):
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

    if selected_slot is None:
        parsed_requested_date = parse_requested_date(
            user_message,
        )

    previous_date = state.get("requested_date")

    if parsed_requested_date is not None:
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

            if state.get("intent") == "book_appointment":
                customer_name = extract_customer_name(
                    user_message=user_message,
                    phone_number=phone_candidate,
                )

                if customer_name is not None:
                    extracted["customer_name"] = customer_name

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
    }:
        return {}

    user_message = get_latest_user_message(state)
    services = get_active_services()
    updates: AppointmentAgentState = {
        "available_services": services,
    }

    allow_numeric_service_choice = (
        state.get("service_id") is None
        and not state.get("available_slots")
    )

    matched_service = find_service_from_message(
        user_message=user_message,
        services=services,
        allow_numeric_choice=allow_numeric_service_choice,
    )

    if matched_service is not None:
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

        if (
            previous_service_id is not None
            and previous_service_id != new_service_id
        ):
            updates["requested_date"] = None

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
        user_message=user_message,
        staff_members=staff_members,
    )

    if matched_staff is not None:
        updates["staff_id"] = int(matched_staff["id"])
        updates["staff_name"] = str(matched_staff["full_name"])

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

    if is_upcoming_availability_request(
        get_latest_user_message(state),
    ):
        if service_id is None:
            return {}

        slots = find_upcoming_available_slots(
            service_id=int(service_id),
            staff_id=(
                int(state["staff_id"])
                if state.get("staff_id") is not None
                else None
            ),
        )

        return {
            "requested_date": None,
            "available_slots": slots,
            "upcoming_alternative_slots": None,
            "slot_id": None,
            "selected_slot_summary": None,
            "booking_summary": None,
            "confirmation_status": "not_requested",
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

    slots = check_available_slots.run(
        tool_arguments,
    )

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

    selected_slot = find_slot_from_message(
        user_message=get_latest_user_message(state),
        available_slots=slots,
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
        }

    return {
        "available_slots": slots,
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

    if state.get("messages") and not user_message.strip():
        return {
            "next_question": (
                "Please type a message so I can help with your "
                "appointment."
            ),
        }

    nlu_result = classify_message(user_message)

    if nlu_result.intent == "ask_notification_capability":
        return {
            "intent": "general_question",
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
                "intent": "general_question",
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
            "intent": "general_question",
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
                        if "dental" in str(
                            service.get("name", "")
                        ).lower()
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
                        if "dermatology" in str(
                            service.get("name", "")
                        ).lower()
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
                        if "physio" in str(
                            service.get("name", "")
                        ).lower()
                    ),
                    None,
                )

        if matched_service is None:
            return {
                "intent": "general_question",
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
            "intent": "general_question",
            "next_question": (
                f"{matched_service['name']} costs {price_text} and "
                f"takes {matched_service.get('duration_minutes')} minutes."
            ),
        }

    if nlu_result.intent == "ask_service_list":
        services = state.get("available_services") or get_active_services()

        return {
            "intent": "general_question",
            "next_question": (
                "These are the services currently available:\n\n"
                + format_service_options(services)
                + "\n\nWould you like to book one of these or check availability?"
            ),
        }
    if nlu_result.intent in {"ask_opening_hours", "ask_location"}:
        return {
            "intent": "general_question",
            "next_question": (
                "I don't have that information available yet. Please "
                "contact the front desk or clinic staff for accurate "
                "details."
            ),
        }
    if nlu_result.intent in {
        "ask_insurance",
        "ask_cancellation_policy",
        "ask_payment_methods",
    }:
        return {
            "intent": "general_question",
            "next_question": (
                "I don't have that information available yet. Please "
                "contact the front desk or clinic staff for accurate "
                "details."
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

    if state.get("slot_selection_error") is not None:
        return {
            "next_question": state["slot_selection_error"],
        }

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
            "next_question": (
                f"{matched_service['name']} costs {price_text} and "
                f"takes {matched_service.get('duration_minutes')} minutes."
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

        else:
            next_question = (
                "Sure, I can check that. Please share your phone number "
                "or appointment reference."
            )

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
                SystemMessage(content=SYSTEM_PROMPT),
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

