from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import Customer
from app.schemas import CustomerCreate, CustomerUpdate


class CustomerAlreadyExistsError(Exception):
    """Raised when a customer with the same phone number or email exists."""


class CustomerNotFoundError(Exception):
    """Raised when a customer cannot be found."""


def create_customer(
    database: Session,
    customer_data: CustomerCreate,
) -> Customer:
    """Create a new customer."""

    duplicate_conditions = [
        Customer.phone_number == customer_data.phone_number,
    ]

    if customer_data.email is not None:
        duplicate_conditions.append(
            Customer.email == customer_data.email
        )

    existing_customer = database.scalar(
        select(Customer).where(
            or_(*duplicate_conditions)
        )
    )

    if existing_customer is not None:
        raise CustomerAlreadyExistsError(
            "A customer with this phone number or email already exists."
        )

    customer = Customer(
        full_name=customer_data.full_name,
        phone_number=customer_data.phone_number,
        email=customer_data.email,
        date_of_birth=customer_data.date_of_birth,
        gender=customer_data.gender,
    )

    database.add(customer)
    database.commit()
    database.refresh(customer)

    return customer


def list_customers(
    database: Session,
    include_inactive: bool = False,
) -> list[Customer]:
    """Return customers, optionally including inactive records."""

    statement = select(Customer).order_by(
        Customer.full_name
    )

    if not include_inactive:
        statement = statement.where(
            Customer.is_active.is_(True)
        )

    return list(
        database.scalars(statement).all()
    )


def get_customer_by_id(
    database: Session,
    customer_id: int,
) -> Customer:
    """Return one customer by ID."""

    customer = database.get(
        Customer,
        customer_id,
    )

    if customer is None:
        raise CustomerNotFoundError(
            f"Customer {customer_id} was not found."
        )

    return customer


def update_customer(
    database: Session,
    customer_id: int,
    customer_data: CustomerUpdate,
) -> Customer:
    """Update selected customer fields."""

    customer = get_customer_by_id(
        database=database,
        customer_id=customer_id,
    )

    update_values = customer_data.model_dump(
        exclude_unset=True
    )

    duplicate_conditions = []

    if "phone_number" in update_values:
        duplicate_conditions.append(
            Customer.phone_number == update_values["phone_number"]
        )

    if (
        "email" in update_values
        and update_values["email"] is not None
    ):
        duplicate_conditions.append(
            Customer.email == update_values["email"]
        )

    if duplicate_conditions:
        duplicate_customer = database.scalar(
            select(Customer).where(
                or_(*duplicate_conditions),
                Customer.id != customer_id,
            )
        )

        if duplicate_customer is not None:
            raise CustomerAlreadyExistsError(
                "A customer with this phone number or email already exists."
            )

    for field_name, field_value in update_values.items():
        setattr(
            customer,
            field_name,
            field_value,
        )

    database.commit()
    database.refresh(customer)

    return customer


def deactivate_customer(
    database: Session,
    customer_id: int,
) -> Customer:
    """Deactivate a customer without deleting appointment history."""

    customer = get_customer_by_id(
        database=database,
        customer_id=customer_id,
    )

    customer.is_active = False

    database.commit()
    database.refresh(customer)

    return customer