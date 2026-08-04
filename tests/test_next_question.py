from app.ai.agent import determine_next_question


def test_asks_for_service_with_available_services() -> None:
    result = determine_next_question(
        {
            "intent": "book_appointment",
            "available_services": [
                {
                    "id": 2,
                    "name": "Dental care",
                    "description": "Dental checkups.",
                    "duration_minutes": 30,
                    "price": 3500,
                }
            ],
        }
    )

    response = result["next_question"]

    assert response is not None
    assert "Sure, I can help you book an appointment." in response
    assert "Which service would you like?" in response
    assert "Available services:" in response
    assert "Dental care" in response


def test_asks_for_date_after_service_is_known() -> None:
    result = determine_next_question(
        {
            "intent": "book_appointment",
            "service_id": 2,
            "service_name": "Dental care",
        }
    )

    response = result["next_question"]

    assert response is not None
    assert "Sure, I can help you book Dental care." in response
    assert "Which date would you prefer?" in response
    assert "tomorrow" in response
    assert "next Monday" in response


def test_handles_no_available_slots() -> None:
    result = determine_next_question(
        {
            "intent": "book_appointment",
            "service_id": 2,
            "service_name": "Dental care",
            "requested_date": "2026-08-05",
            "available_slots": [],
        }
    )

    response = result["next_question"]

    assert response is not None
    assert "There are no available Dental care slots" in response
    assert "Wednesday, 05 August 2026" in response
    assert "Which other date would you prefer?" in response


def test_shows_friendly_available_slots() -> None:
    result = determine_next_question(
        {
            "intent": "book_appointment",
            "service_id": 2,
            "service_name": "Dental care",
            "requested_date": "2026-08-05",
            "available_slots": [
                {
                    "slot_id": 7,
                    "service_id": 2,
                    "service_name": "Dental care",
                    "staff_id": 5,
                    "staff_name": "Dr. Perera",
                    "start_datetime": "2026-08-05T10:00:00",
                    "end_datetime": "2026-08-05T10:30:00",
                    "status": "AVAILABLE",
                }
            ],
        }
    )

    response = result["next_question"]

    assert response is not None
    assert "I found these Dental care slots" in response
    assert "Wednesday, 05 August 2026" in response
    assert "1. 10:00 AM – 10:30 AM with Dr. Perera" in response
    assert "Which one would you prefer?" in response


def test_asks_for_customer_details_after_slot_selection() -> None:
    result = determine_next_question(
        {
            "intent": "book_appointment",
            "service_id": 2,
            "service_name": "Dental care",
            "requested_date": "2026-08-05",
            "slot_id": 7,
            "selected_slot_summary": (
                "Dental care at 10:00 AM with Dr. Perera"
            ),
        }
    )

    response = result["next_question"]

    assert response is not None
    assert (
        "Great. You selected Dental care at 10:00 AM "
        "with Dr. Perera."
    ) in response
    assert "full name and phone number" in response


def test_check_availability_asks_for_service() -> None:
    result = determine_next_question(
        {
            "intent": "check_availability",
            "available_services": [
                {
                    "id": 2,
                    "name": "Dental care",
                    "description": "Dental checkups.",
                    "duration_minutes": 30,
                    "price": 3500,
                }
            ],
        }
    )

    response = result["next_question"]

    assert response is not None
    assert "Which service would you like to check?" in response
    assert "Available services:" in response
    assert "Dental care" in response


def test_check_availability_asks_for_date() -> None:
    result = determine_next_question(
        {
            "intent": "check_availability",
            "service_id": 2,
            "service_name": "Dental care",
        }
    )

    response = result["next_question"]

    assert response == (
        "Which date would you like to check for Dental care?"
    )


def test_cancel_asks_for_appointment_id() -> None:
    result = determine_next_question(
        {
            "intent": "cancel_appointment",
        }
    )

    assert result["next_question"] == (
        "Sure, I can help cancel an appointment. "
        "What is your appointment ID?"
    )


def test_cancel_asks_for_reason() -> None:
    result = determine_next_question(
        {
            "intent": "cancel_appointment",
            "appointment_id": 12,
        }
    )

    assert result["next_question"] == (
        "What is the reason for the cancellation?"
    )


def test_reschedule_asks_for_appointment_id() -> None:
    result = determine_next_question(
        {
            "intent": "reschedule_appointment",
        }
    )

    assert result["next_question"] == (
    "Sure, I can help reschedule an appointment. "
    "What is your appointment ID or reference number?"
)


def test_reschedule_asks_for_new_slot() -> None:
    result = determine_next_question(
        {
            "intent": "reschedule_appointment",
            "appointment_id": 12,
        }
    )

    assert result["next_question"] == (
    "Which date would you like to move this appointment to?"
)


def test_general_question_has_no_controlled_question() -> None:
    result = determine_next_question(
        {
            "intent": "general_question",
        }
    )

    assert result["next_question"] is None