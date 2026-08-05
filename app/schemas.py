from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models import AppointmentStatus, AvailabilityStatus

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
    services: list[ServiceResponse] = Field(default_factory=list)

    model_config = ConfigDict(
        from_attributes=True,
    )


class StaffServiceAssignment(BaseModel):
    service_id: int


class AvailabilitySlotCreate(BaseModel):
    """Information required to create an availability slot."""

    staff_id: int = Field(gt=0)
    service_id: int = Field(gt=0)
    start_datetime: datetime
    end_datetime: datetime


class AvailabilitySlotUpdate(BaseModel):
    """Fields that may be changed on an availability slot."""

    start_datetime: datetime | None = None
    end_datetime: datetime | None = None
    status: AvailabilityStatus | None = None


class AvailabilitySlotResponse(BaseModel):
    """Availability slot returned by the API."""

    id: int
    staff_id: int
    service_id: int
    start_datetime: datetime
    end_datetime: datetime
    status: AvailabilityStatus

    model_config = ConfigDict(from_attributes=True)

class CustomerCreate(BaseModel):
    full_name: str = Field(
        min_length=2,
        max_length=150,
    )

    phone_number: str = Field(
        min_length=7,
        max_length=30,
    )

    email: EmailStr | None = None
    date_of_birth: date | None = None

    gender: str | None = Field(
        default=None,
        max_length=20,
    )


class CustomerUpdate(BaseModel):
    full_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=150,
    )

    phone_number: str | None = Field(
        default=None,
        min_length=7,
        max_length=30,
    )

    email: EmailStr | None = None
    date_of_birth: date | None = None

    gender: str | None = Field(
        default=None,
        max_length=20,
    )

    is_active: bool | None = None


class CustomerResponse(BaseModel):
    id: int
    full_name: str
    phone_number: str
    email: EmailStr | None
    date_of_birth: date | None
    gender: str | None
    is_active: bool

    model_config = ConfigDict(
        from_attributes=True,
    )

class AppointmentCreate(BaseModel):
    customer_id: int = Field(gt=0)
    service_id: int = Field(gt=0)
    staff_id: int = Field(gt=0)
    slot_id: int = Field(gt=0)

    customer_notes: str | None = Field(
        default=None,
        max_length=1000,
    )


class AppointmentCancel(BaseModel):
    cancellation_reason: str = Field(
        min_length=2,
        max_length=1000,
    )


class AppointmentReschedule(BaseModel):
    new_slot_id: int = Field(gt=0)


class AppointmentResponse(BaseModel):
    id: int
    reference_number: str
    customer_id: int
    service_id: int
    staff_id: int
    slot_id: int
    start_datetime: datetime
    end_datetime: datetime
    status: AppointmentStatus
    customer_notes: str | None
    cancellation_reason: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )
class AIChatRequest(BaseModel):
    """One message sent to the appointment agent."""

    message: str = Field(
        min_length=1,
        max_length=4000,
    )

    thread_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )


class AIChatOption(BaseModel):
    """A safe selectable option for browser and future voice clients."""

    id: int
    label: str
    start_datetime: str | None = None
    end_datetime: str | None = None


class AIChatResponse(BaseModel):
    """Response returned by the appointment agent."""

    thread_id: str
    response: str
    message: str
    intent: str | None = None
    conversation_stage: str
    requires_confirmation: bool = False
    pending_action: str | None = None
    options: list[AIChatOption] = Field(default_factory=list)
    error: str | None = None
