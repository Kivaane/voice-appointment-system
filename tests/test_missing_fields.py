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
                "customer_id",
                "service_id",
                "staff_id",
                "slot_id",
            ],
        ),
        (
            {
                "intent": "book_appointment",
                "customer_id": 1,
                "service_id": 2,
            },
            [
                "staff_id",
                "slot_id",
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