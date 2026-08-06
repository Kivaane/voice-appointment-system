"""Message matching and text normalization helpers."""

import re

from app.ai.nlu import normalize_domain_typos


def normalize_text(text: str) -> str:
    """Normalize text for simple matching."""

    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        normalize_domain_typos(text).lower(),
    ).strip()
    return re.sub(r"\b([ap])\s+m\b", r"\1m", normalized)


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

