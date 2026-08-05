import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import SessionLocal
from app.time_utils import business_today
from app.models import (
    Appointment,
    AppointmentStatus,
    AvailabilitySlot,
    AvailabilityStatus,
    Customer,
    Service,
    Staff,
)


DEMO_SERVICE_NAMES = [
    "Dental care",
    "General consultation",
    "Dermatology",
    "Physiotherapy",
]

LEGACY_SERVICE_NAMES = [
    "Dental Consultation",
    "Physiotherapy Session",
]

DEMO_STAFF_EMAILS = [
    "perera@example.com",
    "silva@example.com",
    "fernando@example.com",
    "nimal@example.com",
]

DEMO_CUSTOMER_PHONE = "0772000001"


def get_or_create_service(
    database,
    name: str,
    description: str,
    duration_minutes: int,
    price: Decimal,
) -> Service:
    service = (
        database.query(Service)
        .filter(Service.name == name)
        .first()
    )

    if service is not None:
        service.description = description
        service.duration_minutes = duration_minutes
        service.price = price
        service.is_active = True
        database.flush()
        return service

    service = Service(
        name=name,
        description=description,
        duration_minutes=duration_minutes,
        price=price,
        is_active=True,
    )

    database.add(service)
    database.flush()

    return service


def get_or_create_staff(
    database,
    full_name: str,
    email: str,
    phone_number: str,
    speciality: str,
) -> Staff:
    staff = (
        database.query(Staff)
        .filter(Staff.email == email)
        .first()
    )

    if staff is not None:
        staff.full_name = full_name
        staff.phone_number = phone_number
        staff.speciality = speciality
        staff.is_active = True
        database.flush()
        return staff

    staff = Staff(
        full_name=full_name,
        email=email,
        phone_number=phone_number,
        speciality=speciality,
        is_active=True,
    )

    database.add(staff)
    database.flush()

    return staff


def get_or_create_customer(
    database,
    full_name: str,
    phone_number: str,
    email: str | None = None,
) -> Customer:
    customer = (
        database.query(Customer)
        .filter(Customer.phone_number == phone_number)
        .first()
    )

    if customer is not None:
        customer.full_name = full_name
        customer.email = email
        customer.is_active = True
        database.flush()
        return customer

    customer = Customer(
        full_name=full_name,
        phone_number=phone_number,
        email=email,
        is_active=True,
    )

    database.add(customer)
    database.flush()

    return customer


def assign_service_to_staff(
    staff: Staff,
    service: Service,
) -> None:
    if service not in staff.services:
        staff.services.append(service)


def deactivate_legacy_services(database) -> None:
    legacy_services = (
        database.query(Service)
        .filter(Service.name.in_(LEGACY_SERVICE_NAMES))
        .all()
    )

    for service in legacy_services:
        service.is_active = False


def delete_demo_appointments_and_slots(
    database,
    service_ids: list[int],
    staff_ids: list[int],
) -> None:
    """Clear demo bookings and slots before recreating fresh demo data."""

    database.query(Appointment).delete(
        synchronize_session=False,
    )

    database.query(AvailabilitySlot).delete(
        synchronize_session=False,
    )

    database.flush()


def create_slot(
    database,
    staff: Staff,
    service: Service,
    start_datetime: datetime,
    duration_minutes: int,
    status: AvailabilityStatus,
) -> AvailabilitySlot:
    slot = AvailabilitySlot(
        staff_id=staff.id,
        service_id=service.id,
        start_datetime=start_datetime,
        end_datetime=start_datetime + timedelta(
            minutes=duration_minutes,
        ),
        status=status,
    )

    database.add(slot)
    database.flush()

    return slot


def create_demo_booked_appointment(
    database,
    customer: Customer,
    service: Service,
    staff: Staff,
    slot: AvailabilitySlot,
) -> Appointment:
    slot.status = AvailabilityStatus.BOOKED

    appointment = Appointment(
        reference_number="DEMO-BOOKED-001",
        customer_id=customer.id,
        service_id=service.id,
        staff_id=staff.id,
        slot_id=slot.id,
        start_datetime=slot.start_datetime,
        end_datetime=slot.end_datetime,
        status=AppointmentStatus.CONFIRMED,
        customer_notes="Demo booked appointment.",
    )

    database.add(appointment)
    database.flush()

    return appointment


def main() -> None:
    database = SessionLocal()

    try:
        dental = get_or_create_service(
            database=database,
            name="Dental care",
            description="Dental checkups and treatment appointments.",
            duration_minutes=30,
            price=Decimal("3500.00"),
        )

        general = get_or_create_service(
            database=database,
            name="General consultation",
            description="General medical consultation appointment.",
            duration_minutes=20,
            price=Decimal("2500.00"),
        )

        dermatology = get_or_create_service(
            database=database,
            name="Dermatology",
            description="Skin care consultation appointment.",
            duration_minutes=30,
            price=Decimal("4000.00"),
        )

        physiotherapy = get_or_create_service(
            database=database,
            name="Physiotherapy",
            description="Physiotherapy treatment appointment.",
            duration_minutes=45,
            price=Decimal("4500.00"),
        )

        dr_perera = get_or_create_staff(
            database=database,
            full_name="Dr. Perera",
            email="perera@example.com",
            phone_number="0771000001",
            speciality="Dental care",
        )

        dr_silva = get_or_create_staff(
            database=database,
            full_name="Dr. Silva",
            email="silva@example.com",
            phone_number="0771000002",
            speciality="General medicine",
        )

        dr_fernando = get_or_create_staff(
            database=database,
            full_name="Dr. Fernando",
            email="fernando@example.com",
            phone_number="0771000003",
            speciality="Dermatology",
        )

        therapist_nimal = get_or_create_staff(
            database=database,
            full_name="Therapist Nimal",
            email="nimal@example.com",
            phone_number="0771000004",
            speciality="Physiotherapy",
        )

        assign_service_to_staff(dr_perera, dental)
        assign_service_to_staff(dr_silva, general)
        assign_service_to_staff(dr_fernando, dermatology)
        assign_service_to_staff(therapist_nimal, physiotherapy)

        customer = get_or_create_customer(
            database=database,
            full_name="Demo Customer",
            phone_number=DEMO_CUSTOMER_PHONE,
            email="demo.customer@example.com",
        )

        deactivate_legacy_services(database)

        service_ids = [
            dental.id,
            general.id,
            dermatology.id,
            physiotherapy.id,
        ]

        staff_ids = [
            dr_perera.id,
            dr_silva.id,
            dr_fernando.id,
            therapist_nimal.id,
        ]

        delete_demo_appointments_and_slots(
            database=database,
            service_ids=service_ids,
            staff_ids=staff_ids,
        )

        today = business_today()
        tomorrow = today + timedelta(days=1)
        day_after_tomorrow = today + timedelta(days=2)
        third_day = today + timedelta(days=3)

        create_slot(
            database=database,
            staff=dr_perera,
            service=dental,
            start_datetime=datetime.combine(
                tomorrow,
                datetime.strptime("10:00", "%H:%M").time(),
            ),
            duration_minutes=dental.duration_minutes,
            status=AvailabilityStatus.AVAILABLE,
        )

        create_slot(
            database=database,
            staff=dr_perera,
            service=dental,
            start_datetime=datetime.combine(
                tomorrow,
                datetime.strptime("14:30", "%H:%M").time(),
            ),
            duration_minutes=dental.duration_minutes,
            status=AvailabilityStatus.AVAILABLE,
        )

        create_slot(
            database=database,
            staff=dr_perera,
            service=dental,
            start_datetime=datetime.combine(
                tomorrow,
                datetime.strptime("16:00", "%H:%M").time(),
            ),
            duration_minutes=dental.duration_minutes,
            status=AvailabilityStatus.BLOCKED,
        )

        dental_booked_slot = create_slot(
            database=database,
            staff=dr_perera,
            service=dental,
            start_datetime=datetime.combine(
                day_after_tomorrow,
                datetime.strptime("09:30", "%H:%M").time(),
            ),
            duration_minutes=dental.duration_minutes,
            status=AvailabilityStatus.BOOKED,
        )

        create_demo_booked_appointment(
            database=database,
            customer=customer,
            service=dental,
            staff=dr_perera,
            slot=dental_booked_slot,
        )

        create_slot(
            database=database,
            staff=dr_silva,
            service=general,
            start_datetime=datetime.combine(
                tomorrow,
                datetime.strptime("11:00", "%H:%M").time(),
            ),
            duration_minutes=general.duration_minutes,
            status=AvailabilityStatus.AVAILABLE,
        )

        create_slot(
            database=database,
            staff=dr_fernando,
            service=dermatology,
            start_datetime=datetime.combine(
                day_after_tomorrow,
                datetime.strptime("15:00", "%H:%M").time(),
            ),
            duration_minutes=dermatology.duration_minutes,
            status=AvailabilityStatus.AVAILABLE,
        )

        create_slot(
            database=database,
            staff=therapist_nimal,
            service=physiotherapy,
            start_datetime=datetime.combine(
                third_day,
                datetime.strptime("10:30", "%H:%M").time(),
            ),
            duration_minutes=physiotherapy.duration_minutes,
            status=AvailabilityStatus.AVAILABLE,
        )

        database.commit()

        print("Demo data reset successfully.")
        print("Dental care tomorrow: 10:00 and 14:30 AVAILABLE.")
        print("Dental care tomorrow: 16:00 BLOCKED.")
        print("Dental care day after tomorrow: 09:30 BOOKED.")

    except Exception:
        database.rollback()
        raise

    finally:
        database.close()


if __name__ == "__main__":
    main()
