from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


from fastapi import FastAPI

from app import models
from app.database import Base, engine
from app.routes.services import router as services_router
from app.routes.staff import router as staff_router

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="AI Voice Appointment System",
    description="Backend API for appointment management.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(services_router)
app.include_router(services_router)
app.include_router(staff_router)

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