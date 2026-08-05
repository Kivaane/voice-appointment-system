from datetime import date

import pytest
from langchain_core.messages import HumanMessage

from app.ai import agent
from app.ai.nlu import classify_message, normalize_domain_typos


SERVICES = [
    {"id": 1, "name": "Dental care", "duration_minutes": 30, "price": 3500},
    {"id": 2, "name": "General consultation", "duration_minutes": 20, "price": 2500},
    {"id": 3, "name": "Dermatology", "duration_minutes": 30, "price": 4000},
    {"id": 4, "name": "Physiotherapy", "duration_minutes": 45, "price": 4500},
]


def state_for(message: str, **values):
    return {
        "messages": [HumanMessage(content=message)],
        "intent": None,
        "confirmation_status": "not_requested",
        **values,
    }


@pytest.mark.parametrize(
    "message",
    [
        "speak to a person",
        "talk to a person",
        "speak to a human",
        "talk to a human",
        "human please",
        "receptionist",
        "front desk",
        "staff member",
        "connect me to someone",
    ],
)
def test_handoff_is_controlled_and_never_calls_semantic_model(
    monkeypatch, message
) -> None:
    monkeypatch.setattr(
        agent,
        "classify_unknown_message",
        lambda _: pytest.fail("handoff must bypass semantic NLU"),
    )

    result = agent.detect_intent(state_for(message))

    assert result["intent"] == "general_question"
    assert result["next_question"] == (
        "I can hand this over to a human staff member. Please contact "
        "the front desk or clinic staff to continue this request."
    )


def test_information_only_does_not_resume_a_transaction() -> None:
    state = state_for(
        "how much is dental",
        intent="book_appointment",
        service_id=1,
        service_name="Dental care",
        transaction_started_explicitly=False,
    )

    assert agent.compose_informational_response("Dental costs LKR 3,500.", state) == (
        "Dental costs LKR 3,500."
    )


def test_information_interruption_resumes_a_real_booking() -> None:
    state = state_for(
        "how much is dental",
        intent="book_appointment",
        service_id=1,
        service_name="Dental care",
        requested_date="2026-08-06",
        available_slots=[
            {
                "slot_id": 1,
                "start_datetime": "2026-08-06T10:00:00",
                "end_datetime": "2026-08-06T10:30:00",
            }
        ],
        transaction_started_explicitly=True,
    )

    response = agent.compose_informational_response("Dental costs LKR 3,500.", state)

    assert response.startswith("Dental costs LKR 3,500.")
    assert "10:00 AM" in response


@pytest.mark.parametrize(
    ("current", "message", "expected"),
    [
        ("reschedule_appointment", "I don't want my appointment anymore", "cancel_appointment"),
        ("book_appointment", "I can't come for my appointment", "cancel_appointment"),
        ("cancel_appointment", "I need to reschedule my appointment", "reschedule_appointment"),
    ],
)
def test_explicit_transaction_intent_switches_immediately(
    current, message, expected
) -> None:
    result = agent.detect_intent(
        state_for(
            message,
            intent=current,
            appointment_id=9,
            appointment_reference_number="APT-SAFE123",
            booking_summary="pending mutation",
            confirmation_status="pending",
            transaction_started_explicitly=True,
        )
    )

    assert result["intent"] == expected
    assert result["appointment_id"] == 9
    assert result["booking_summary"] is None
    assert result["confirmation_status"] == "not_requested"


def test_status_temporarily_pauses_rescheduling() -> None:
    result = agent.detect_intent(
        state_for(
            "What time is my appointment?",
            intent="reschedule_appointment",
            appointment_id=9,
            transaction_started_explicitly=True,
        )
    )

    assert result["intent"] == "view_appointments"
    assert result["paused_intent"] == "reschedule_appointment"


@pytest.mark.parametrize(
    ("message", "corrected"),
    [
        ("phydyotherapy", "physiotherapy"),
        ("appointemnt", "appointment"),
        ("appoinment", "appointment"),
        ("tomorrorw", "tomorrow"),
        ("availab", "available"),
        ("surger", "surgery"),
        ("ypur", "your"),
    ],
)
def test_controlled_domain_typo_normalization(message, corrected) -> None:
    assert normalize_domain_typos(message) == corrected


def test_typo_normalization_preserves_phone_and_reference() -> None:
    original = "appointemnt APT-ABC123 0774588691 tomorrorw"
    corrected = normalize_domain_typos(original)

    assert "APT-ABC123" in corrected
    assert "0774588691" in corrected
    assert classify_message("cancel my appointemnt").intent == "cancel_appointment"
    assert classify_message("do you do surger").intent == "ask_service_availability"


def test_service_switch_keeps_new_date_and_clears_dependent_choices(monkeypatch) -> None:
    monkeypatch.setattr(agent, "business_today", lambda: date(2026, 8, 5))
    monkeypatch.setattr(agent, "get_active_services", lambda: SERVICES)
    monkeypatch.setattr(
        agent,
        "get_active_staff_for_service",
        lambda service_id: [{"id": 44, "full_name": "Therapist Nimal"}],
    )
    base = state_for(
        "any physio slot next Monday",
        intent="check_availability",
        service_id=1,
        service_name="Dental care",
        requested_date="2026-08-06",
        available_slots=[{"slot_id": 10}],
        slot_id=10,
        transaction_started_explicitly=True,
    )
    extracted = agent.extract_details(base)
    merged = {**base, **extracted}
    resolved = agent.resolve_named_entities(merged)

    assert extracted["requested_date"] == "2026-08-10"
    assert resolved["service_id"] == 4
    assert resolved["available_slots"] is None
    assert "requested_date" not in resolved


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Monday", "2026-08-10"),
        ("next Monday", "2026-08-10"),
        ("this Friday", "2026-08-07"),
        ("on the 6th", "2026-08-06"),
        ("tomorrorw", "2026-08-06"),
    ],
)
def test_date_parsing_uses_nearest_upcoming_date(monkeypatch, message, expected) -> None:
    monkeypatch.setattr(agent, "business_today", lambda: date(2026, 8, 5))
    assert agent.parse_requested_date(message) == expected


def test_vague_availability_clears_stale_date_and_results() -> None:
    result = agent.detect_intent(
        state_for(
            "when is available",
            intent="check_availability",
            service_id=1,
            requested_date="2026-08-17",
            available_slots=[{"slot_id": 10}],
            transaction_started_explicitly=True,
        )
    )

    assert result["requested_date"] is None
    assert result["available_slots"] is None


def make_slots():
    return [
        {"slot_id": 1, "start_datetime": "2026-08-06T10:00:00"},
        {"slot_id": 2, "start_datetime": "2026-08-06T14:30:00"},
        {"slot_id": 3, "start_datetime": "2026-08-06T17:30:00"},
    ]


@pytest.mark.parametrize(
    ("message", "expected_ids"),
    [
        ("morning slots", [1]),
        ("afternoon slots", [2]),
        ("evening slots", [3]),
        ("after 2 pm", [2, 3]),
        ("after 2 p.m.", [2, 3]),
        ("after two pm", [2, 3]),
        ("before noon", [1]),
        ("earliest available slot", [1]),
        ("latest available slot", [3]),
    ],
)
def test_time_preferences_filter_without_mutating_source(message, expected_ids) -> None:
    slots = make_slots()
    original = list(slots)
    preference, error = agent.extract_time_preference(message)

    assert error is None
    filtered = agent.filter_slots_by_time_preference(slots, preference)
    assert [slot["slot_id"] for slot in filtered] == expected_ids
    assert slots == original


@pytest.mark.parametrize("choice", ["second", "2nd", "2"])
def test_service_option_accepts_safe_ordinals(choice) -> None:
    assert agent.find_service_from_message(choice, SERVICES, True)["id"] == 2


def test_new_booking_resets_all_transactional_fields_but_preserves_customer() -> None:
    result = agent.detect_intent(
        state_for(
            "I want to book another appointment",
            intent="book_appointment",
            customer_id=7,
            customer_name="Kivaane Anton",
            customer_phone_number="+94774588691",
            service_id=2,
            requested_date="2026-09-02",
            slot_id=22,
            appointment_id=99,
            appointment_reference_number="APT-OLD123",
            confirmation_status="confirmed",
        )
    )

    assert result["intent"] == "book_appointment"
    for field in (
        "service_id",
        "staff_id",
        "requested_date",
        "time_preference",
        "available_slots",
        "slot_id",
        "appointment_id",
        "appointment_reference_number",
        "booking_summary",
    ):
        assert result[field] is None
    assert "customer_name" not in result
    assert "customer_phone_number" not in result


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("book_appointment", "already confirmed"),
        ("cancel_appointment", "already cancelled"),
        ("reschedule_appointment", "already been rescheduled"),
    ],
)
def test_repeated_confirmation_returns_idempotent_controlled_message(intent, expected) -> None:
    result = agent.detect_intent(
        state_for(
            "yes",
            intent=intent,
            appointment_id=9,
            appointment_reference_number="APT-SAFE123",
            confirmation_status="confirmed",
        )
    )

    assert expected in result["next_question"]
    assert "APT-SAFE123" in result["next_question"]


@pytest.mark.parametrize(
    "message",
    [
        "Book physiotherapy next Monday",
        "I need dental tomorrow. My name is Kivaane Anton and my number is 0774588691.",
        "Book physiotherapy next Monday afternoon for Kivaane Anton, phone 0774588691.",
        "Dental with Dr. Perera tomorrow morning",
    ],
)
def test_observed_one_message_bookings_extract_service_and_date(
    monkeypatch, message
) -> None:
    monkeypatch.setattr(agent, "business_today", lambda: date(2026, 8, 5))
    monkeypatch.setattr(agent, "get_active_services", lambda: SERVICES)
    monkeypatch.setattr(
        agent,
        "get_active_staff_for_service",
        lambda service_id: [
            {
                "id": 11 if service_id == 1 else 44,
                "full_name": "Dr. Perera" if service_id == 1 else "Therapist Nimal",
            }
        ],
    )
    base = state_for(message)
    detected = agent.detect_intent(base)
    extracted = agent.extract_details({**base, **detected})
    resolved = agent.resolve_named_entities({**base, **detected, **extracted})

    assert detected["intent"] == "book_appointment"
    assert extracted["requested_date"] in {"2026-08-06", "2026-08-10"}
    assert resolved["service_id"] in {1, 4}
