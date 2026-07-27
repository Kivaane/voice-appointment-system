from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import EmailStr

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


class StaffCreate(BaseModel):
    full_name: str = Field(
        min_length=2,
        max_length=150,
    )

    email: EmailStr

    phone_number: str | None = Field(
        default=None,
        min_length=7,
        max_length=30,
    )

    speciality: str | None = Field(
        default=None,
        max_length=150,
    )


class StaffUpdate(BaseModel):
    full_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )

    email: EmailStr | None = None

    phone_number: str | None = Field(
        default=None,
        min_length=7,
        max_length=30,
    )

    speciality: str | None = Field(
        default=None,
        max_length=150,
    )

    is_active: bool | None = None


class StaffResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    phone_number: str | None
    speciality: str | None
    is_active: bool
    services: list[ServiceResponse] = []

    model_config = ConfigDict(
        from_attributes=True,
    )


class StaffServiceAssignment(BaseModel):
    service_id: int