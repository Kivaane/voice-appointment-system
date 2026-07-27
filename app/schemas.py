from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class ServiceCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    duration_minutes: int = Field(ge=5, le=480)
    price: Decimal | None = Field(default=None, ge=0)


class ServiceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    duration_minutes: int | None = Field(default=None, ge=5, le=480)
    price: Decimal | None = Field(default=None, ge=0)
    is_active: bool | None = None


class ServiceResponse(BaseModel):
    id: int
    name: str
    description: str | None
    duration_minutes: int
    price: Decimal | None
    is_active: bool

    model_config = ConfigDict(from_attributes=True)