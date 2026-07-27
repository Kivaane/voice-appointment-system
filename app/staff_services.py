from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import Service, Staff
from app.schemas import StaffCreate, StaffUpdate


class StaffAlreadyExistsError(Exception):
    """Raised when a staff member with the same email already exists."""


class StaffNotFoundError(Exception):
    """Raised when a staff member cannot be found."""


class ServiceAssignmentError(Exception):
    """Raised when assigning an invalid or duplicate service."""


def create_staff(
    database: Session,
    staff_data: StaffCreate,
) -> Staff:
    """Create a new staff member."""

    existing_staff = database.scalar(
        select(Staff).where(
            Staff.email == staff_data.email
        )
    )

    if existing_staff is not None:
        raise StaffAlreadyExistsError(
            "A staff member with this email already exists."
        )

    staff = Staff(
        full_name=staff_data.full_name,
        email=staff_data.email,
        phone_number=staff_data.phone_number,
        speciality=staff_data.speciality,
    )

    database.add(staff)
    database.commit()
    database.refresh(staff)

    return staff


def list_staff(
    database: Session,
    include_inactive: bool = False,
) -> list[Staff]:
    """Return staff members with their assigned services."""

    statement = (
        select(Staff)
        .options(selectinload(Staff.services))
        .order_by(Staff.full_name)
    )

    if not include_inactive:
        statement = statement.where(
            Staff.is_active.is_(True)
        )

    return list(
        database.scalars(statement).all()
    )


def get_staff_by_id(
    database: Session,
    staff_id: int,
) -> Staff:
    """Return one staff member with assigned services."""

    statement = (
        select(Staff)
        .options(selectinload(Staff.services))
        .where(Staff.id == staff_id)
    )

    staff = database.scalar(statement)

    if staff is None:
        raise StaffNotFoundError(
            f"Staff member {staff_id} was not found."
        )

    return staff


def update_staff(
    database: Session,
    staff_id: int,
    staff_data: StaffUpdate,
) -> Staff:
    """Update selected staff fields."""

    staff = get_staff_by_id(
        database=database,
        staff_id=staff_id,
    )

    update_values = staff_data.model_dump(
        exclude_unset=True
    )

    if "email" in update_values:
        duplicate_staff = database.scalar(
            select(Staff).where(
                Staff.email == update_values["email"],
                Staff.id != staff_id,
            )
        )

        if duplicate_staff is not None:
            raise StaffAlreadyExistsError(
                "A staff member with this email already exists."
            )

    for field_name, field_value in update_values.items():
        setattr(staff, field_name, field_value)

    database.commit()

    return get_staff_by_id(
        database=database,
        staff_id=staff_id,
    )


def deactivate_staff(
    database: Session,
    staff_id: int,
) -> Staff:
    """Deactivate a staff member without deleting history."""

    staff = get_staff_by_id(
        database=database,
        staff_id=staff_id,
    )

    staff.is_active = False

    database.commit()

    return get_staff_by_id(
        database=database,
        staff_id=staff_id,
    )


def assign_service_to_staff(
    database: Session,
    staff_id: int,
    service_id: int,
) -> Staff:
    """Assign one active service to a staff member."""

    staff = get_staff_by_id(
        database=database,
        staff_id=staff_id,
    )

    service = database.get(Service, service_id)

    if service is None:
        raise ServiceAssignmentError(
            f"Service {service_id} was not found."
        )

    if not service.is_active:
        raise ServiceAssignmentError(
            "Inactive services cannot be assigned."
        )

    if service in staff.services:
        raise ServiceAssignmentError(
            "This service is already assigned to the staff member."
        )

    staff.services.append(service)

    database.commit()

    return get_staff_by_id(
        database=database,
        staff_id=staff_id,
    )


def remove_service_from_staff(
    database: Session,
    staff_id: int,
    service_id: int,
) -> Staff:
    """Remove one service assignment from a staff member."""

    staff = get_staff_by_id(
        database=database,
        staff_id=staff_id,
    )

    service = next(
        (
            item
            for item in staff.services
            if item.id == service_id
        ),
        None,
    )

    if service is None:
        raise ServiceAssignmentError(
            "This service is not assigned to the staff member."
        )

    staff.services.remove(service)

    database.commit()

    return get_staff_by_id(
        database=database,
        staff_id=staff_id,
    )