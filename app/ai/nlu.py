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
    if is_service_availability_question(text):
        return NLUResult(
            intent="ask_service_availability",
            confidence=0.9,
            should_start_booking=False,
        )

    return NLUResult(intent="unknown", confidence=0.3)