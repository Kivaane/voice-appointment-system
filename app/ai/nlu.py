import re
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field


@dataclass
class NLUResult:
    intent: str
    confidence: float = 1.0
    service_hint: Optional[str] = None
    date_hint: Optional[str] = None
    time_hint: Optional[str] = None
    customer_name: Optional[str] = None
    phone_hint: Optional[str] = None
    appointment_reference: Optional[str] = None
    should_start_booking: bool = False
    staff_hint: Optional[str] = None
    requires_clarification: bool = False
    clarification_reason: Optional[str] = None


class NLUModelResult(BaseModel):
    """Validated, classification-only output from the semantic fallback."""

    intent: Literal[
        "book_appointment",
        "cancel_appointment",
        "reschedule_appointment",
        "check_availability",
        "view_appointments",
        "list_services",
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
        "general_question",
        "unknown",
    ]
    confidence: float = Field(ge=0, le=1)
    service_hint: str | None = None
    staff_hint: str | None = None
    date_hint: str | None = None
    time_hint: str | None = None
    customer_name: str | None = None
    phone_hint: str | None = None
    appointment_reference: str | None = None
    requires_clarification: bool = False
    clarification_reason: str | None = None


DOMAIN_TYPO_CORRECTIONS = {
    "phydyotherapy": "physiotherapy",
    "physiotheraphy": "physiotherapy",
    "appoinment": "appointment",
    "appointemnt": "appointment",
    "appoinmtent": "appointment",
    "appointmentt": "appointment",
    "tomorrorw": "tomorrow",
    "tommorow": "tomorrow",
    "avalable": "available",
    "availab": "available",
    "surger": "surgery",
    "ypur": "your",
    "reshedule": "reschedule",
}


def normalize_domain_typos(message: str) -> str:
    """Correct only known appointment-domain typos, preserving other text."""

    corrected = message
    for typo, replacement in DOMAIN_TYPO_CORRECTIONS.items():
        corrected = re.sub(
            rf"\b{re.escape(typo)}\b",
            replacement,
            corrected,
            flags=re.IGNORECASE,
        )
    return corrected


def normalize_message(message: str) -> str:
    corrected = normalize_domain_typos(message)
    return " ".join(corrected.lower().strip().split())


def is_notification_question(text: str) -> bool:
    notification_phrases = [
        "will you notify me",
        "do you notify me",
        "notify me",
        "notify please",
        "notification",
        "notifications",
        "will i get a reminder",
        "will i get reminder",
        "do you send reminders",
        "send reminders",
        "will you remind me",
        "appointment reminder",
        "reminder",
        "sms reminder",
        "email reminder",
        "do you send sms",
        "do you send email",
        "will i get sms",
        "will i get email",
    ]

    if any(phrase in text for phrase in notification_phrases):
        return True

    has_booking_context = (
        "book" in text
        or "appointment" in text
        or "booking" in text
        or "slot" in text
    )

    has_notification_context = (
        "notify" in text
        or "notification" in text
        or "remind" in text
        or "reminder" in text
        or "sms" in text
        or "email" in text
    )

    return has_booking_context and has_notification_context

def is_service_availability_question(text: str) -> bool:
    service_question_phrases = [
        "do you have",
        "do you offer",
        "do you provide",
        "do you do",
        "is there",
        "are there",
        "available service",
        "service available",
        "can i get",
        "can you do",
    ]

    service_words = [
        "dental",
        "dentist",
        "tooth",
        "teeth",
        "physio",
        "physiotherapy",
        "dermatology",
        "skin",
        "general consultation",
        "consultation",
        "surgery",
        "cleaning",
    ]

    return any(phrase in text for phrase in service_question_phrases) and any(
        word in text for word in service_words
    )

def is_pricing_question(text: str) -> bool:
    pricing_words = [
        "price",
        "prices",
        "cost",
        "costs",
        "fee",
        "fees",
        "charge",
        "charges",
        "how much",
        "rate",
        "rates",
    ]

    service_words = [
        "dental",
        "dentist",
        "tooth",
        "teeth",
        "physio",
        "physiotherapy",
        "dermatology",
        "skin",
        "general consultation",
        "consultation",
    ]

    return any(word in text for word in pricing_words) and (
        any(word in text for word in service_words)
        or "service" in text
        or "appointment" in text
    )


def is_service_list_question(text: str) -> bool:
    service_list_phrases = [
        "what services do you have",
        "what services are available",
        "show services",
        "list services",
        "available services",
        "services available",
        "what do you offer",
        "what can i book",
        "what appointments can i book",
        "what kind of appointments",
    ]

    return any(phrase in text for phrase in service_list_phrases)

def is_opening_hours_question(text: str) -> bool:
    opening_hours_phrases = [
        "opening hours",
        "open hours",
        "working hours",
        "business hours",
        "what time are you open",
        "when are you open",
        "are you open",
        "what time do you close",
        "when do you close",
        "open today",
        "open tomorrow",
        "clinic hours",
    ]

    return any(phrase in text for phrase in opening_hours_phrases)


def is_location_question(text: str) -> bool:
    location_phrases = [
        "where are you located",
        "where is the clinic",
        "clinic location",
        "your location",
        "location",
        "address",
        "where are you",
        "how do i get there",
        "directions",
    ]

    return any(phrase in text for phrase in location_phrases)

def is_insurance_question(text: str) -> bool:
    insurance_phrases = [
        "insurance",
        "do you accept insurance",
        "accept insurance",
        "medical insurance",
        "health insurance",
        "claim insurance",
        "insurance claim",
        "can i use insurance",
    ]

    return any(phrase in text for phrase in insurance_phrases)


def is_cancellation_policy_question(text: str) -> bool:
    cancellation_policy_phrases = [
        "cancellation policy",
        "cancel policy",
        "policy for cancellation",
        "can i cancel",
        "how late can i cancel",
        "cancel fee",
        "cancellation fee",
        "reschedule policy",
        "policy for rescheduling",
    ]

    return any(phrase in text for phrase in cancellation_policy_phrases)


def is_payment_methods_question(text: str) -> bool:
    payment_phrases = [
        "payment",
        "payments",
        "payment methods",
        "how can i pay",
        "can i pay by card",
        "card payment",
        "cash payment",
        "online payment",
        "do you accept card",
        "do you take cash",
    ]

    return any(phrase in text for phrase in payment_phrases)
def is_natural_booking_request(text: str) -> bool:
    booking_need_phrases = [
        "book",
        "schedule",
        "i need",
        "need",
        "i want",
        "want",
        "can i get",
        "can i make",
        "can i schedule",
        "can you schedule",
        "i would like",
        "looking for",
    ]

    service_words = [
        "dentist",
        "dental",
        "tooth",
        "teeth",
        "doctor",
        "consultation",
        "physio",
        "physiotherapy",
        "skin",
        "dermatology",
        "appointment",
        "slot",
    ]

    question_only_phrases = [
        "do you have",
        "do you offer",
        "do you provide",
        "how much",
        "price",
        "cost",
        "what services",
        "which services",
        "will you notify",
        "notify me",
        "reminder",
        "insurance",
        "payment",
        "location",
        "opening hours",
    ]

    if any(phrase in text for phrase in question_only_phrases):
        return False

    blocked_action_phrases = [
        "reschedule",
        "cancel",
        "move my appointment",
        "move appointment",
        "change my appointment",
        "change appointment",
        "change the date",
        "change the time",
        "change my slot",
        "different date",
        "different time",
        "i don't want",
        "i dont want",
        "i do not want",
        "i can't come",
        "i cant come",
        "not coming",
    ]

    if any(phrase in text for phrase in blocked_action_phrases):
        return False

    has_service = any(word in text for word in service_words)
    has_booking_phrase = any(
        phrase in text for phrase in booking_need_phrases
    )
    has_direct_date = any(
        word in text
        for word in (
            "today",
            "tomorrow",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        )
    )

    return has_service and (has_booking_phrase or has_direct_date)


def is_natural_reschedule_request(text: str) -> bool:
    reschedule_phrases = [
        "reschedule",
        "reschedule my appointment",
        "need to reschedule",
        "i need to reschedule",
        "want to reschedule",
        "i want to reschedule",
        "move my appointment",
        "move appointment",
        "move it",
        "change my appointment",
        "change appointment",
        "change the date",
        "change the time",
        "change my slot",
        "change slot",
        "move my slot",
        "shift my appointment",
        "shift appointment",
        "can i move",
        "can i change",
        "another time",
        "different time",
        "different date",
    ]

    booked_context_words = [
        "appointment",
        "slot",
        "booking",
        "it",
        "time",
        "date",
    ]

    return any(phrase in text for phrase in reschedule_phrases) and any(
        word in text for word in booked_context_words
    )


def is_natural_cancellation_request(text: str) -> bool:
    cancellation_phrases = [
        "cancel my appointment",
        "cancel appointment",
        "cancel my booking",
        "cancel booking",
        "i want to cancel",
        "i need to cancel",
        "i don't want my appointment",
        "i dont want my appointment",
        "i do not want my appointment",
        "i don't need my appointment",
        "i dont need my appointment",
        "i do not need my appointment",
        "remove my appointment",
        "delete my appointment",
        "drop my appointment",
        "stop my appointment",
        "i cannot come",
        "i can't come",
        "i cant come",
        "i will not come",
        "i won't come",
        "i wont come",
        "unable to attend",
        "not coming",
    ]

    return any(phrase in text for phrase in cancellation_phrases)

def is_natural_availability_request(text: str) -> bool:
    availability_words = [
        "available",
        "availability",
        "slot",
        "slots",
        "free slot",
        "free slots",
        "free time",
        "any time",
        "any slots",
        "when can i come",
    ]

    date_or_time_words = [
        "today",
        "tomorrow",
        "day after tomorrow",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "morning",
        "afternoon",
        "evening",
        "next",
    ]

    service_words = [
        "dental",
        "dentist",
        "tooth",
        "teeth",
        "physio",
        "physiotherapy",
        "dermatology",
        "skin",
        "general consultation",
        "consultation",
    ]

    if "available services" in text or "services available" in text:
        return False

    if (
        any(word in text for word in ("slot", "slots", "time", "times"))
        and any(word in text for word in ("available", "availability"))
    ):
        return True

    return any(word in text for word in availability_words) and (
        any(word in text for word in date_or_time_words)
        or any(word in text for word in service_words)
    )


def is_appointment_status_question(text: str) -> bool:
    status_phrases = [
        "do i have any appointments",
        "do i have appointment",
        "my appointments",
        "my bookings",
        "appointment status",
        "booking status",
        "show my appointment",
        "show my appointments",
        "view my appointment",
        "view my appointments",
        "what time is my appointment",
        "when is my appointment",
    ]

    return any(phrase in text for phrase in status_phrases)


def is_duration_question(text: str) -> bool:
    duration_words = [
        "how long",
        "duration",
        "how many minutes",
        "how much time",
        "time does it take",
    ]

    service_words = [
        "dental",
        "dentist",
        "tooth",
        "teeth",
        "physio",
        "physiotherapy",
        "dermatology",
        "skin",
        "general consultation",
        "consultation",
        "service",
        "appointment",
    ]

    return any(word in text for word in duration_words) and any(
        word in text for word in service_words
    )





def classify_message(message: str) -> NLUResult:
    text = normalize_message(message)

    if not text:
        return NLUResult(intent="blank", confidence=1.0)

    if is_notification_question(text):
        return NLUResult(
            intent="ask_notification_capability",
            confidence=0.98,
            should_start_booking=False,
        )
    if is_pricing_question(text):
        return NLUResult(
            intent="ask_pricing",
            confidence=0.92,
            should_start_booking=False,
        )

    if is_service_list_question(text):
        return NLUResult(
            intent="ask_service_list",
            confidence=0.92,
            should_start_booking=False,
        )

    if is_opening_hours_question(text):
        return NLUResult(
            intent="ask_opening_hours",
            confidence=0.92,
            should_start_booking=False,
        )

    if is_location_question(text):
        return NLUResult(
            intent="ask_location",
            confidence=0.92,
            should_start_booking=False,
        )
    if is_insurance_question(text):
        return NLUResult(
            intent="ask_insurance",
            confidence=0.92,
            should_start_booking=False,
        )

    if is_cancellation_policy_question(text):
        return NLUResult(
            intent="ask_cancellation_policy",
            confidence=0.92,
            should_start_booking=False,
        )

    if is_payment_methods_question(text):
        return NLUResult(
            intent="ask_payment_methods",
            confidence=0.92,
            should_start_booking=False,
        )
    if is_natural_cancellation_request(text):
        return NLUResult(
            intent="cancel_appointment",
            confidence=0.88,
            should_start_booking=False,
        )

    if is_natural_reschedule_request(text):
        return NLUResult(
            intent="reschedule_appointment",
            confidence=0.88,
            should_start_booking=False,
        )

    if is_duration_question(text):
        return NLUResult(
            intent="ask_duration",
            confidence=0.9,
            should_start_booking=False,
        )

    if is_natural_availability_request(text):
        return NLUResult(
            intent="check_availability",
            confidence=0.88,
            should_start_booking=False,
        )

    if is_appointment_status_question(text):
        return NLUResult(
            intent="view_appointments",
            confidence=0.9,
            should_start_booking=False,
        )

    if is_natural_booking_request(text):
        return NLUResult(
            intent="book_appointment",
            confidence=0.86,
            should_start_booking=True,
        )
    if is_service_availability_question(text):
        return NLUResult(
            intent="ask_service_availability",
            confidence=0.9,
            should_start_booking=False,
        )

    return NLUResult(intent="unknown", confidence=0.3)
