from datetime import datetime, timedelta
from decimal import Decimal

from app.database import SessionLocal
from app.models import (
    Appointment,
    AppointmentStatus,
    AvailabilitySlot,
    AvailabilityStatus,
    Customer,
    Service,
    Staff,
)


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


def get_or_create_slot(
    database,
    staff: Staff,
    service: Service,
    start_datetime: datetime,
    duration_minutes: int,
    status: AvailabilityStatus,
) -> AvailabilitySlot:
    slot = (
        database.query(AvailabilitySlot)
        .filter(
            AvailabilitySlot.staff_id == staff.id,
            AvailabilitySlot.service_id == service.id,
            AvailabilitySlot.start_datetime == start_datetime,
        )
        .first()
    )

    end_datetime = start_datetime + timedelta(
        minutes=duration_minutes,
    )

    if slot is not None:
        slot.end_datetime = end_datetime
        slot.status = status
        return slot

    slot = AvailabilitySlot(
        staff_id=staff.id,
        service_id=service.id,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        status=status,
    )

    database.add(slot)
    database.flush()

    return slot


def get_or_create_demo_appointment(
    database,
    customer: Customer,
    service: Service,
    staff: Staff,
    slot: AvailabilitySlot,
) -> Appointment:
    appointment = (
        database.query(Appointment)
        .filter(
            Appointment.reference_number
            == "DEMO-BOOKED-001"
        )
        .first()
    )

    slot.status = AvailabilityStatus.BOOKED

    if appointment is not None:
        appointment.customer_id = customer.id
        appointment.service_id = service.id
        appointment.staff_id = staff.id
        appointment.slot_id = slot.id
        appointment.start_datetime = slot.start_datetime
        appointment.end_datetime = slot.end_datetime
        appointment.status = AppointmentStatus.CONFIRMED
        return appointment

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
            phone_number="0772000001",
            email="demo.customer@example.com",
        )

        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        day_after_tomorrow = today + timedelta(days=2)
        third_day = today + timedelta(days=3)

        dental_available_1 = get_or_create_slot(
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

        dental_available_2 = get_or_create_slot(
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

        dental_blocked = get_or_create_slot(
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

        dental_booked = get_or_create_slot(
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

        get_or_create_demo_appointment(
            database=database,
            customer=customer,
            service=dental,
            staff=dr_perera,
            slot=dental_booked,
        )

        get_or_create_slot(
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

        get_or_create_slot(
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

        get_or_create_slot(
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

        print("Demo seed data created successfully.")
        print(f"Dental care service ID: {dental.id}")
        print(f"Dr. Perera staff ID: {dr_perera.id}")
        print(f"Demo customer ID: {customer.id}")
        print(
            "Available dental slots:",
            dental_available_1.id,
            dental_available_2.id,
        )
        print("Blocked dental slot:", dental_blocked.id)
        print("Booked dental slot:", dental_booked.id)

    finally:
        database.close()


if __name__ == "__main__":
    main()