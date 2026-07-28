from fastapi.testclient import TestClient


def test_health_check(
    client: TestClient,
) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_complete_appointment_lifecycle(
    client: TestClient,
) -> None:
    service_response = client.post(
        "/services",
        json={
            "name": "Sports Therapy Test",
            "description": "Automated test service.",
            "duration_minutes": 45,
            "price": 5000,
        },
    )

    assert service_response.status_code == 201
    service_id = service_response.json()["id"]

    staff_response = client.post(
        "/staff",
        json={
            "full_name": "Test Therapist",
            "email": "test.therapist@example.com",
            "phone_number": "0711111111",
            "speciality": "Sports Therapy",
        },
    )

    assert staff_response.status_code == 201
    staff_id = staff_response.json()["id"]

    assignment_response = client.post(
        f"/staff/{staff_id}/assign-service",
        json={
            "service_id": service_id,
        },
    )

    assert assignment_response.status_code == 200

    first_slot_response = client.post(
        "/availability",
        json={
            "staff_id": staff_id,
            "service_id": service_id,
            "start_datetime": "2026-08-01T09:00:00",
            "end_datetime": "2026-08-01T09:45:00",
        },
    )

    assert first_slot_response.status_code == 201
    first_slot_id = first_slot_response.json()["id"]

    second_slot_response = client.post(
        "/availability",
        json={
            "staff_id": staff_id,
            "service_id": service_id,
            "start_datetime": "2026-08-01T10:00:00",
            "end_datetime": "2026-08-01T10:45:00",
        },
    )

    assert second_slot_response.status_code == 201
    second_slot_id = second_slot_response.json()["id"]

    customer_response = client.post(
        "/customers",
        json={
            "full_name": "Test Customer",
            "phone_number": "0761111111",
            "email": "test.customer@example.com",
            "date_of_birth": "2002-05-10",
            "gender": "Female",
        },
    )

    assert customer_response.status_code == 201
    customer_id = customer_response.json()["id"]

    appointment_response = client.post(
        "/appointments",
        json={
            "customer_id": customer_id,
            "service_id": service_id,
            "staff_id": staff_id,
            "slot_id": first_slot_id,
            "customer_notes": "Automated lifecycle test",
        },
    )

    assert appointment_response.status_code == 201

    appointment_data = appointment_response.json()
    appointment_id = appointment_data["id"]

    assert appointment_data["status"] == "CONFIRMED"
    assert appointment_data["slot_id"] == first_slot_id

    first_slot_check = client.get(
        f"/availability/{first_slot_id}"
    )

    assert first_slot_check.status_code == 200
    assert first_slot_check.json()["status"] == "BOOKED"

    duplicate_booking_response = client.post(
        "/appointments",
        json={
            "customer_id": customer_id,
            "service_id": service_id,
            "staff_id": staff_id,
            "slot_id": first_slot_id,
            "customer_notes": "Duplicate booking attempt",
        },
    )

    assert duplicate_booking_response.status_code == 409

    reschedule_response = client.post(
        f"/appointments/{appointment_id}/reschedule",
        json={
            "new_slot_id": second_slot_id,
        },
    )

    assert reschedule_response.status_code == 200

    rescheduled_data = reschedule_response.json()

    assert rescheduled_data["slot_id"] == second_slot_id
    assert rescheduled_data["status"] == "CONFIRMED"

    old_slot_check = client.get(
        f"/availability/{first_slot_id}"
    )

    new_slot_check = client.get(
        f"/availability/{second_slot_id}"
    )

    assert old_slot_check.json()["status"] == "AVAILABLE"
    assert new_slot_check.json()["status"] == "BOOKED"

    cancellation_response = client.post(
        f"/appointments/{appointment_id}/cancel",
        json={
            "cancellation_reason": "Automated cancellation test",
        },
    )

    assert cancellation_response.status_code == 200
    assert (
        cancellation_response.json()["status"]
        == "CANCELLED_BY_CUSTOMER"
    )

    released_slot_check = client.get(
        f"/availability/{second_slot_id}"
    )

    assert released_slot_check.json()["status"] == "AVAILABLE"