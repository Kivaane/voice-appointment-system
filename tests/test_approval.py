from app.ai.approval import (
    resume_booking_approval,
    start_booking_approval,
)


def test_booking_approval_can_be_approved() -> None:
    thread_id = "approval-thread-001"

    paused_result = start_booking_approval(
        thread_id=thread_id,
        booking_details={
            "service": "Dental care",
            "date": "2026-08-01",
            "time": "10:00",
        },
    )

    assert "__interrupt__" in paused_result

    completed_result = resume_booking_approval(
        thread_id=thread_id,
        approved=True,
    )

    assert completed_result["approval_status"] == "approved"


def test_booking_approval_can_be_rejected() -> None:
    thread_id = "approval-thread-002"

    paused_result = start_booking_approval(
        thread_id=thread_id,
        booking_details={
            "service": "General consultation",
            "date": "2026-08-02",
            "time": "14:00",
        },
    )

    assert "__interrupt__" in paused_result

    completed_result = resume_booking_approval(
        thread_id=thread_id,
        approved=False,
    )

    assert completed_result["approval_status"] == "rejected"