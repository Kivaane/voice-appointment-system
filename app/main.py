import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from app.middleware import RequestLoggingMiddleware
from fastapi import FastAPI

from app.logging_config import configure_logging
from app.routes.appointments import router as appointments_router
from app.routes.availability import router as availability_router
from app.routes.customers import router as customers_router
from app.routes.services import router as services_router
from app.routes.staff import router as staff_router
from app.routes.ai_chat import router as ai_chat_router

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Log application startup and shutdown events."""

    logger.info("Application starting")
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title="AI Voice Appointment System",
    description="Backend API for appointment management.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(services_router)
app.include_router(staff_router)
app.include_router(availability_router)
app.include_router(customers_router)
app.include_router(appointments_router)
app.add_middleware(RequestLoggingMiddleware)
app.include_router(ai_chat_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "AI Voice Appointment System API",
        "status": "running",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
    }