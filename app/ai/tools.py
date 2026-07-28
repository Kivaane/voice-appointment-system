from langchain.tools import tool

from app.database import SessionLocal
from app.services import list_services
from app.config import get_settings

@tool
def list_available_services() -> list[dict[str, object]]:
    """Return all active appointment services offered by the business.

    Use this tool when a customer asks what services are available,
    what they can book, or which appointment types are offered.
    """

    database = SessionLocal()
    settings = get_settings()

    try:
        services = list_services(
            database=database,
            include_inactive=False,
        )

        return [
    {
        "id": service.id,
        "name": service.name,
        "description": service.description,
        "duration_minutes": service.duration_minutes,
        "price": (
            float(service.price)
            if service.price is not None
            else None
        ),
        "currency": settings.currency,
    }
    for service in services
]

    finally:
        database.close()