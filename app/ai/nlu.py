from dataclasses import dataclass
from typing import Optional


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


def normalize_message(message: str) -> str:
    return " ".join(message.lower().strip().split())


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
    if is_service_availability_question(text):
        return NLUResult(
            intent="ask_service_availability",
            confidence=0.9,
            should_start_booking=False,
        )

    return NLUResult(intent="unknown", confidence=0.3)