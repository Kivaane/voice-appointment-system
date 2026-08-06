"""Customer identification and phone number normalisation helpers.

All functions in this module are pure text-processing utilities with no
database access.  Database-backed customer resolution remains in agent.py
so that test fixtures that monkeypatch ``app.ai.agent.SessionLocal`` continue
to work without modification.
"""

import re


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
        r"\b(?:my\s+name\s+is|name\s+is|i\s+am|i['\u2019]m|this\s+is)"
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
