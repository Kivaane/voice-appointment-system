from datetime import datetime
from decimal import Decimal
from enum import Enum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SqlEnum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Table,
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


class Staff(Base):
    """A staff member who can provide appointment services."""

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