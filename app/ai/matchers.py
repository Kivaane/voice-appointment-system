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
