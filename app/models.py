from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Enum as SqlEnum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


staff_services = Table(
    "staff_services",
    Base.metadata,
    Column(
        "staff_id",
        ForeignKey("staff.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "service_id",
        ForeignKey("services.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class AvailabilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    HELD = "HELD"
    BOOKED = "BOOKED"
    BLOCKED = "BLOCKED"


class AppointmentStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    CANCELLED_BY_CUSTOMER = "CANCELLED_BY_CUSTOMER"
    CANCELLED_BY_BUSINESS = "CANCELLED_BY_BUSINESS"
    RESCHEDULING_REQUIRED = "RESCHEDULING_REQUIRED"
    COMPLETED = "COMPLETED"
    NO_SHOW = "NO_SHOW"


class Service(Base):
    """A service that customers can book."""

    __tablename__ = "services"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    duration_minutes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    price: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 2),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    staff_members: Mapped[list["Staff"]] = relationship(
        secondary=staff_services,
        back_populates="services",
    )

    availability_slots: Mapped[list["AvailabilitySlot"]] = relationship(
        back_populates="service",
    )

    appointments: Mapped[list["Appointment"]] = relationship(
        back_populates="service",
    )


class Staff(Base):
    """A staff member who provides appointment services."""

    __tablename__ = "staff"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    full_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )

    phone_number: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    speciality: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    services: Mapped[list[Service]] = relationship(
        secondary=staff_services,
        back_populates="staff_members",
    )

    availability_slots: Mapped[list["AvailabilitySlot"]] = relationship(
        back_populates="staff",
    )

    appointments: Mapped[list["Appointment"]] = relationship(
        back_populates="staff",
    )


class AvailabilitySlot(Base):
    """A bookable time period for one staff member and service."""

    __tablename__ = "availability_slots"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    staff_id: Mapped[int] = mapped_column(
        ForeignKey("staff.id"),
        nullable=False,
        index=True,
    )

    service_id: Mapped[int] = mapped_column(
        ForeignKey("services.id"),
        nullable=False,
        index=True,
    )

    start_datetime: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        index=True,
    )

    end_datetime: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )

    status: Mapped[AvailabilityStatus] = mapped_column(
        SqlEnum(AvailabilityStatus),
        default=AvailabilityStatus.AVAILABLE,
        nullable=False,
        index=True,
    )

    staff: Mapped["Staff"] = relationship(
        back_populates="availability_slots",
    )

    service: Mapped["Service"] = relationship(
        back_populates="availability_slots",
    )

    appointment: Mapped["Appointment | None"] = relationship(
        back_populates="slot",
        uselist=False,
    )


class Customer(Base):
    """A customer who books appointments."""

    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    full_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    phone_number: Mapped[str] = mapped_column(
        String(30),
        unique=True,
        nullable=False,
    )

    email: Mapped[str | None] = mapped_column(
        String(255),
        unique=True,
        nullable=True,
    )

    date_of_birth: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )

    gender: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    appointments: Mapped[list["Appointment"]] = relationship(
        back_populates="customer",
    )


class Appointment(Base):
    """A confirmed booking connecting all appointment entities."""

    __tablename__ = "appointments"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    reference_number: Mapped[str] = mapped_column(
        String(30),
        unique=True,
        nullable=False,
        index=True,
    )

    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id"),
        nullable=False,
        index=True,
    )

    service_id: Mapped[int] = mapped_column(
        ForeignKey("services.id"),
        nullable=False,
        index=True,
    )

    staff_id: Mapped[int] = mapped_column(
        ForeignKey("staff.id"),
        nullable=False,
        index=True,
    )

    slot_id: Mapped[int] = mapped_column(
        ForeignKey("availability_slots.id"),
        unique=True,
        nullable=False,
        index=True,
    )

    start_datetime: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        index=True,
    )

    end_datetime: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )

    status: Mapped[AppointmentStatus] = mapped_column(
        SqlEnum(AppointmentStatus),
        default=AppointmentStatus.CONFIRMED,
        nullable=False,
        index=True,
    )

    customer_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    cancellation_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    customer: Mapped[Customer] = relationship(
        back_populates="appointments",
    )

    service: Mapped[Service] = relationship(
        back_populates="appointments",
    )

    staff: Mapped[Staff] = relationship(
        back_populates="appointments",
    )

    slot: Mapped[AvailabilitySlot] = relationship(
        back_populates="appointment",
    )