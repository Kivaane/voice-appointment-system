from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Service
from app.schemas import ServiceCreate, ServiceUpdate


class ServiceAlreadyExistsError(Exception):
    """Raised when a service with the same name already exists."""


class ServiceNotFoundError(Exception):
    """Raised when a requested service cannot be found."""


def create_service(
    database: Session,
    service_data: ServiceCreate,
) -> Service:
    """Create a new bookable service."""

    existing_service = database.scalar(
        select(Service).where(
            Service.name == service_data.name
        )
    )

    if existing_service is not None:
        raise ServiceAlreadyExistsError(
            "A service with this name already exists."
        )

    service = Service(
        name=service_data.name,
        description=service_data.description,
        duration_minutes=service_data.duration_minutes,
        price=service_data.price,
    )

    database.add(service)
    database.commit()
    database.refresh(service)

    return service


def list_services(
    database: Session,
    include_inactive: bool = False,
) -> list[Service]:
    """Return services, optionally including inactive records."""

    statement = select(Service).order_by(Service.name)

    if not include_inactive:
        statement = statement.where(
            Service.is_active.is_(True)
        )

    return list(
        database.scalars(statement).all()
    )


def get_service_by_id(
    database: Session,
    service_id: int,
) -> Service:
    """Return one service by its identifier."""

    service = database.get(Service, service_id)

    if service is None:
        raise ServiceNotFoundError(
            f"Service {service_id} was not found."
        )

    return service


def update_service(
    database: Session,
    service_id: int,
    service_data: ServiceUpdate,
) -> Service:
    """Update selected fields of an existing service."""

    service = get_service_by_id(
        database=database,
        service_id=service_id,
    )

    update_values = service_data.model_dump(
        exclude_unset=True
    )

    if "name" in update_values:
        duplicate_service = database.scalar(
            select(Service).where(
                Service.name == update_values["name"],
                Service.id != service_id,
            )
        )

        if duplicate_service is not None:
            raise ServiceAlreadyExistsError(
                "A service with this name already exists."
            )

    for field_name, field_value in update_values.items():
        setattr(service, field_name, field_value)

    database.commit()
    database.refresh(service)

    return service


def deactivate_service(
    database: Session,
    service_id: int,
) -> Service:
    """Deactivate a service without deleting its history."""

    service = get_service_by_id(
        database=database,
        service_id=service_id,
    )

    service.is_active = False

    database.commit()
    database.refresh(service)

    return service