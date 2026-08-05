import pytest

from app.ai.agent import calculate_missing_fields


@pytest.mark.parametrize(
    ("state", "expected_missing_fields"),
    [
        (
            {
                "intent": "book_appointment",
            },
            [
                "service_id",
                "requested_date",
                "slot_id",
                "customer_id",
            ],
        ),
        (
            {
                "intent": "book_appointment",
                "service_id": 2,
            },
            [
                "requested_date",
                "slot_id",
                "customer_id",
            ],
        ),
        (
            {
                "intent": "book_appointment",
                "service_id": 2,
                "requested_date": "2026-08-05",
                "slot_id": 7,
            },
            [
                "customer_id",
            ],
        ),
        (
            {
                "intent": "check_availability",
                "service_id": 2,
            },
            [
                "requested_date",
            ],
        ),
        (
            {
                "intent": "cancel_appointment",
                "appointment_id": 10,
            },
            [
                "cancellation_reason",
            ],
        ),
        (
            {
                "intent": "reschedule_appointment",
                "appointment_id": 10,
                "slot_id": 20,
            },
            [],
        ),
        (
            {
                "intent": "general_question",
            },
            [],
        ),
    ],
)
def test_calculates_missing_fields(
    state,
    expected_missing_fields,
) -> None:
    result = calculate_missing_fields(state)

    assert result["missing_fields"] == expected_missing_fields