from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum

from sqlalchemy import (
    JSON,
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
class AIConversationStatus(str, Enum):
    ACTIVE = "ACTIVE"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AIMessageRole(str, Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    TOOL = "TOOL"


class AIEventType(str, Enum):
    MODEL_REQUEST = "MODEL_REQUEST"
    MODEL_RESPONSE = "MODEL_RESPONSE"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    STATE_CHANGE = "STATE_CHANGE"
    HUMAN_APPROVAL_REQUESTED = "HUMAN_APPROVAL_REQUESTED"
    HUMAN_APPROVED = "HUMAN_APPROVED"
    HUMAN_REJECTED = "HUMAN_REJECTED"
    ERROR = "ERROR"

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
    ai_conversations: Mapped[list["AIConversation"]] = relationship(
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
class AIConversation(Base):
    """A persistent AI conversation identified by a thread ID."""

    __tablename__ = "ai_conversations"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    thread_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    customer_id: Mapped[int | None] = mapped_column(
        ForeignKey("customers.id"),
        nullable=True,
        index=True,
    )

    status: Mapped[AIConversationStatus] = mapped_column(
        SqlEnum(AIConversationStatus),
        default=AIConversationStatus.ACTIVE,
        nullable=False,
        index=True,
    )

    current_intent: Mapped[str | None] = mapped_column(
        String(100),
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

    customer: Mapped["Customer | None"] = relationship(
        back_populates="ai_conversations",
    )

    messages: Mapped[list["AIMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )

    events: Mapped[list["AIEvent"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


class AIMessage(Base):
    """A message exchanged during an AI conversation."""

    __tablename__ = "ai_messages"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("ai_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    role: Mapped[AIMessageRole] = mapped_column(
        SqlEnum(AIMessageRole),
        nullable=False,
        index=True,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    model_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    conversation: Mapped["AIConversation"] = relationship(
        back_populates="messages",
    )


class AIEvent(Base):
    """A structured event produced during an AI agent run."""

    __tablename__ = "ai_events"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("ai_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    request_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    event_type: Mapped[AIEventType] = mapped_column(
        SqlEnum(AIEventType),
        nullable=False,
        index=True,
    )

    event_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    event_data: Mapped[dict | list | None] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    conversation: Mapped["AIConversation"] = relationship(
        back_populates="events",
    )
