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
    InvalidAppointmentError,
    create_appointment,
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
from app.models import AIEventType, AIMessageRole, Customer, Service, Staff
from app.schemas import AppointmentCreate

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
    service_id: int | None
    service_name: str | None
    staff_id: int | None
    staff_name: str | None
    requested_date: str | None

    available_services: list[dict[str, object]] | None
    available_slots: list[dict[str, object]] | None
    selected_slot_summary: str | None
    booking_summary: str | None
    slot_id: int | None

    appointment_id: int | None
    appointment_reference_number: str | None
    cancellation_reason: str | None

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

            for service in services:
                if service["id"] == number:
                    return service

            if 1 <= number <= len(services):
                return services[number - 1]

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
    InvalidAppointmentError,
    create_appointment,
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
from app.models import AIEventType, AIMessageRole, Customer, Service, Staff
from app.schemas import AppointmentCreate

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
    service_id: int | None
    service_name: str | None
    staff_id: int | None
    staff_name: str | None
    requested_date: str | None

    available_services: list[dict[str, object]] | None
    available_slots: list[dict[str, object]] | None
    selected_slot_summary: str | None
    booking_summary: str | None
    slot_id: int | None

    appointment_id: int | None
    appointment_reference_number: str | None
    cancellation_reason: str | None

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

            for service in services:
                if service["id"] == number:
                    return service

            if 1 <= number <= len(services):
                return services[number - 1]

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


def find_slot_from_message(
    user_message: str,
    available_slots: list[dict[str, object]] | None,
) -> dict[str, object] | None:
    """Resolve a chosen slot from natural text."""

    if not available_slots:
        return None

    normalized_message = normalize_text(user_message)

    slot_id_match = re.search(
        r"\bslot(?:\s+id)?\s*(?:is|=|:)?\s*(\d+)\b",
        user_message,
        flags=re.IGNORECASE,
    )

    if slot_id_match is not None:
        selected_slot_id = int(slot_id_match.group(1))

        for slot in available_slots:
            if slot.get("slot_id") == selected_slot_id:
                return slot

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
        if word in normalized_message:
            if 1 <= option_number <= len(available_slots):
                return available_slots[option_number - 1]

    bare_number_match = re.fullmatch(
        r"\s*(\d+)\s*\.?\s*",
        user_message,
    )

    if bare_number_match is not None:
        number = int(bare_number_match.group(1))

        for slot in available_slots:
            if slot.get("slot_id") == number:
                return slot

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
            f"{hour_12}:{minute}",
            f"{hour_12}:{minute} {start_datetime.strftime('%p').lower()}",
        }

        if any(
            time_text in normalized_message
            for time_text in possible_times
        ):
            return slot

    return None

def extract_phone_number(
    user_message: str,
) -> str | None:
    """Extract a likely phone number from a customer message."""

    phone_match = re.search(
        r"\b(?:\+?\d[\d\s-]{6,}\d)\b",
        user_message,
    )

    if phone_match is None:
        return None

    phone_number = re.sub(
        r"[\s-]+",
        "",
        phone_match.group(0),
    )

    return phone_number


def extract_customer_name(
    user_message: str,
    phone_number: str | None,
) -> str | None:
    """Extract a simple customer name from a message."""

    cleaned_message = user_message.strip()

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

    database = SessionLocal()

    try:
        customer = (
            database.query(Customer)
            .filter(Customer.phone_number == phone_number)
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
            phone_number=phone_number,
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

def is_confirmation_yes_message(
    user_message: str,
) -> bool:
    """Return True when the user confirms a pending booking."""

    normalized_message = normalize_text(user_message)

    yes_phrases = {
        "yes",
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
            "start_datetime": appointment.start_datetime.isoformat(),
            "end_datetime": appointment.end_datetime.isoformat(),
        }

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


def detect_intent(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Identify the user's current appointment intent."""

    user_message = get_latest_user_message(state).lower()

    if (
        state.get("confirmation_status") == "pending"
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
            "appointment",
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

    selected_slot = find_slot_from_message(
        user_message=user_message,
        available_slots=state.get("available_slots"),
    )

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

        if state.get("confirmation_status") == "pending":
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
        match = re.search(
            pattern,
            user_message,
            flags=re.IGNORECASE,
        )

        if match is not None:
            extracted[field_name] = int(match.group(1))

    parsed_date = parse_requested_date(user_message)

    if parsed_date is not None:
        previous_date = state.get("requested_date")

        if previous_date is not None and parsed_date != previous_date:
            extracted["available_slots"] = None
            extracted["slot_id"] = None
            extracted["selected_slot_summary"] = None
            extracted["booking_summary"] = None
            extracted["confirmation_status"] = "not_requested"

        extracted["requested_date"] = parsed_date

    if (
        state.get("confirmation_status") == "pending"
        and selected_slot is None
        and is_time_correction_message(user_message)
    ):
        extracted["slot_id"] = None
        extracted["selected_slot_summary"] = None
        extracted["booking_summary"] = None
        extracted["confirmation_status"] = "not_requested"
        extracted["appointment_id"] = None
        extracted["appointment_reference_number"] = None

    if state.get("intent") == "book_appointment":
        phone_number = extract_phone_number(user_message)

        if phone_number is not None:
            extracted["customer_phone_number"] = phone_number

            customer_name = extract_customer_name(
                user_message=user_message,
                phone_number=phone_number,
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
            updates["slot_id"] = None
            updates["selected_slot_summary"] = None
            updates["staff_id"] = None
            updates["staff_name"] = None

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

    customer = get_or_create_customer_from_details(
        full_name=customer_name,
        phone_number=phone_number,
    )

    return {
        "customer_id": int(customer["id"]),
        "customer_name": str(customer["full_name"]),
        "customer_phone_number": str(customer["phone_number"]),
    }

def confirm_or_reject_booking(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Create the appointment after the user confirms the summary."""

    if state.get("intent") != "book_appointment":
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

    if state.get("confirmation_status") != "confirmed":
        return {}

    required_fields = [
        "customer_id",
        "service_id",
        "staff_id",
        "slot_id",
    ]

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
    }:
        return {}

    service_id = state.get("service_id")
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

    services = state.get("available_services") or get_active_services()

    if (
        intent == "book_appointment"
        and state.get("appointment_id") is not None
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

    if intent == "list_services":
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
            service_name = state.get("service_name") or "appointment"
            friendly_date = format_date(
                state.get("requested_date"),
            )

            if not available_slots:
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
        if state.get("appointment_id") is None:
            next_question = (
                "Sure, I can help cancel an appointment. "
                "What is your appointment ID?"
            )

        elif state.get("cancellation_reason") is None:
            next_question = (
                "What is the reason for the cancellation?"
            )

    elif intent == "reschedule_appointment":
        if state.get("appointment_id") is None:
            next_question = (
                "Sure, I can help reschedule an appointment. "
                "What is your appointment ID?"
            )

        elif state.get("slot_id") is None:
            next_question = (
                "Which new appointment slot would you prefer?"
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


def find_slot_from_message(
    user_message: str,
    available_slots: list[dict[str, object]] | None,
) -> dict[str, object] | None:
    """Resolve a chosen slot from natural text."""

    if not available_slots:
        return None

    normalized_message = normalize_text(user_message)

    slot_id_match = re.search(
        r"\bslot(?:\s+id)?\s*(?:is|=|:)?\s*(\d+)\b",
        user_message,
        flags=re.IGNORECASE,
    )

    if slot_id_match is not None:
        selected_slot_id = int(slot_id_match.group(1))

        for slot in available_slots:
            if slot.get("slot_id") == selected_slot_id:
                return slot

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
        if word in normalized_message:
            if 1 <= option_number <= len(available_slots):
                return available_slots[option_number - 1]

    bare_number_match = re.fullmatch(
        r"\s*(\d+)\s*\.?\s*",
        user_message,
    )

    if bare_number_match is not None:
        number = int(bare_number_match.group(1))

        for slot in available_slots:
            if slot.get("slot_id") == number:
                return slot

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
            f"{hour_12}:{minute}",
            f"{hour_12}:{minute} {start_datetime.strftime('%p').lower()}",
        }

        if any(
            time_text in normalized_message
            for time_text in possible_times
        ):
            return slot

    return None

def extract_phone_number(
    user_message: str,
) -> str | None:
    """Extract a likely phone number from a customer message."""

    phone_match = re.search(
        r"\b(?:\+?\d[\d\s-]{6,}\d)\b",
        user_message,
    )

    if phone_match is None:
        return None

    phone_number = re.sub(
        r"[\s-]+",
        "",
        phone_match.group(0),
    )

    return phone_number


def extract_customer_name(
    user_message: str,
    phone_number: str | None,
) -> str | None:
    """Extract a simple customer name from a message."""

    cleaned_message = user_message.strip()

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

    database = SessionLocal()

    try:
        customer = (
            database.query(Customer)
            .filter(Customer.phone_number == phone_number)
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
            phone_number=phone_number,
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

def is_confirmation_yes_message(
    user_message: str,
) -> bool:
    """Return True when the user confirms a pending booking."""

    normalized_message = normalize_text(user_message)

    yes_phrases = {
        "yes",
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
            "start_datetime": appointment.start_datetime.isoformat(),
            "end_datetime": appointment.end_datetime.isoformat(),
        }

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


def detect_intent(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Identify the user's current appointment intent."""

    user_message = get_latest_user_message(state).lower()

    if (
        state.get("confirmation_status") == "pending"
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
            "appointment",
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

    selected_slot = find_slot_from_message(
        user_message=user_message,
        available_slots=state.get("available_slots"),
    )

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

        if state.get("confirmation_status") == "pending":
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
        match = re.search(
            pattern,
            user_message,
            flags=re.IGNORECASE,
        )

        if match is not None:
            extracted[field_name] = int(match.group(1))

    parsed_date = parse_requested_date(user_message)

    if parsed_date is not None:
        previous_date = state.get("requested_date")

        if previous_date is not None and parsed_date != previous_date:
            extracted["available_slots"] = None
            extracted["slot_id"] = None
            extracted["selected_slot_summary"] = None
            extracted["booking_summary"] = None
            extracted["confirmation_status"] = "not_requested"

        extracted["requested_date"] = parsed_date

    if (
        state.get("confirmation_status") == "pending"
        and selected_slot is None
        and is_time_correction_message(user_message)
    ):
        extracted["slot_id"] = None
        extracted["selected_slot_summary"] = None
        extracted["booking_summary"] = None
        extracted["confirmation_status"] = "not_requested"
        extracted["appointment_id"] = None
        extracted["appointment_reference_number"] = None

    if state.get("intent") == "book_appointment":
        phone_number = extract_phone_number(user_message)

        if phone_number is not None:
            extracted["customer_phone_number"] = phone_number

            customer_name = extract_customer_name(
                user_message=user_message,
                phone_number=phone_number,
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
            updates["slot_id"] = None
            updates["selected_slot_summary"] = None
            updates["staff_id"] = None
            updates["staff_name"] = None

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

    customer = get_or_create_customer_from_details(
        full_name=customer_name,
        phone_number=phone_number,
    )

    return {
        "customer_id": int(customer["id"]),
        "customer_name": str(customer["full_name"]),
        "customer_phone_number": str(customer["phone_number"]),
    }

def confirm_or_reject_booking(
    state: AppointmentAgentState,
) -> AppointmentAgentState:
    """Create the appointment after the user confirms the summary."""

    if state.get("intent") != "book_appointment":
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

    if state.get("confirmation_status") != "confirmed":
        return {}

    required_fields = [
        "customer_id",
        "service_id",
        "staff_id",
        "slot_id",
    ]

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
    }:
        return {}

    service_id = state.get("service_id")
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

    services = state.get("available_services") or get_active_services()

    if (
        intent == "book_appointment"
        and state.get("appointment_id") is not None
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

    if intent == "list_services":
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
            service_name = state.get("service_name") or "appointment"
            friendly_date = format_date(
                state.get("requested_date"),
            )

            if not available_slots:
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
        if state.get("appointment_id") is None:
            next_question = (
                "Sure, I can help cancel an appointment. "
                "What is your appointment ID?"
            )

        elif state.get("cancellation_reason") is None:
            next_question = (
                "What is the reason for the cancellation?"
            )

    elif intent == "reschedule_appointment":
        if state.get("appointment_id") is None:
            next_question = (
                "Sure, I can help reschedule an appointment. "
                "What is your appointment ID?"
            )

        elif state.get("slot_id") is None:
            next_question = (
                "Which new appointment slot would you prefer?"
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