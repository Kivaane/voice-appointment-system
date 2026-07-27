from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.appointment_services import (
    AppointmentConflictError,
    AppointmentNotFoundError,
    InvalidAppointmentError,
    cancel_appointment,
    create_appointment,
    get_appointment_by_id,
    list_appointments,
    reschedule_appointment,
)
from app.database import get_db
from app.models import AppointmentStatus
from app.schemas import (
    AppointmentCancel,
    AppointmentCreate,
    AppointmentReschedule,
    AppointmentResponse,
)

router = APIRouter(
    prefix="/appointments",
    tags=["Appointments"],
)


@router.post(
    "",
    response_model=AppointmentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_new_appointment(
    appointment: AppointmentCreate,
    database: Session = Depends(get_db),
):
    try:
        return create_appointment(
            database,
            appointment,
        )
    except InvalidAppointmentError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )
    except AppointmentConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        )


@router.get(
    "",
    response_model=list[AppointmentResponse],
)
def get_all_appointments(
    customer_id: int | None = Query(default=None, gt=0),
    staff_id: int | None = Query(default=None, gt=0),
    service_id: int | None = Query(default=None, gt=0),
    appointment_status: AppointmentStatus | None = Query(default=None),
    database: Session = Depends(get_db),
):
    return list_appointments(
        database=database,
        customer_id=customer_id,
        staff_id=staff_id,
        service_id=service_id,
        appointment_status=appointment_status,
    )


@router.get(
    "/{appointment_id}",
    response_model=AppointmentResponse,
)
def get_single_appointment(
    appointment_id: int,
    database: Session = Depends(get_db),
):
    try:
        return get_appointment_by_id(
            database,
            appointment_id,
        )
    except AppointmentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )


@router.post(
    "/{appointment_id}/cancel",
    response_model=AppointmentResponse,
)
def cancel_existing_appointment(
    appointment_id: int,
    cancellation: AppointmentCancel,
    database: Session = Depends(get_db),
):
    try:
        return cancel_appointment(
            database=database,
            appointment_id=appointment_id,
            cancellation_reason=cancellation.cancellation_reason,
        )
    except AppointmentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )
    except InvalidAppointmentError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )


@router.post(
    "/{appointment_id}/reschedule",
    response_model=AppointmentResponse,
)
def reschedule_existing_appointment(
    appointment_id: int,
    reschedule: AppointmentReschedule,
    database: Session = Depends(get_db),
):
    try:
        return reschedule_appointment(
            database=database,
            appointment_id=appointment_id,
            new_slot_id=reschedule.new_slot_id,
        )
    except AppointmentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )
    except InvalidAppointmentError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )
    except AppointmentConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        )