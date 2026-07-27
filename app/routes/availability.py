from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.availability_services import (
    AvailabilityConflictError,
    AvailabilitySlotNotFoundError,
    InvalidAvailabilitySlotError,
    activate_availability_slot,
    block_availability_slot,
    create_availability_slot,
    get_availability_slot_by_id,
    list_availability_slots,
    update_availability_slot,
)
from app.database import get_db
from app.models import AvailabilityStatus
from app.schemas import (
    AvailabilitySlotCreate,
    AvailabilitySlotResponse,
    AvailabilitySlotUpdate,
)

router = APIRouter(
    prefix="/availability",
    tags=["Availability"],
)


@router.post(
    "",
    response_model=AvailabilitySlotResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_new_availability_slot(
    slot: AvailabilitySlotCreate,
    database: Session = Depends(get_db),
):
    try:
        return create_availability_slot(
            database,
            slot,
        )
    except AvailabilityConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        )
    except InvalidAvailabilitySlotError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )


@router.get(
    "",
    response_model=list[AvailabilitySlotResponse],
)
def get_all_availability_slots(
    staff_id: int | None = Query(default=None, gt=0),
    service_id: int | None = Query(default=None, gt=0),
    requested_date: date | None = Query(default=None),
    slot_status: AvailabilityStatus | None = Query(default=None),
    database: Session = Depends(get_db),
):
    return list_availability_slots(
        database=database,
        staff_id=staff_id,
        service_id=service_id,
        requested_date=requested_date,
        slot_status=slot_status,
    )


@router.get(
    "/{slot_id}",
    response_model=AvailabilitySlotResponse,
)
def get_single_availability_slot(
    slot_id: int,
    database: Session = Depends(get_db),
):
    try:
        return get_availability_slot_by_id(
            database,
            slot_id,
        )
    except AvailabilitySlotNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )


@router.patch(
    "/{slot_id}",
    response_model=AvailabilitySlotResponse,
)
def edit_availability_slot(
    slot_id: int,
    slot: AvailabilitySlotUpdate,
    database: Session = Depends(get_db),
):
    try:
        return update_availability_slot(
            database,
            slot_id,
            slot,
        )
    except AvailabilitySlotNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )
    except AvailabilityConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        )
    except InvalidAvailabilitySlotError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )


@router.post(
    "/{slot_id}/block",
    response_model=AvailabilitySlotResponse,
)
def block_slot(
    slot_id: int,
    database: Session = Depends(get_db),
):
    try:
        return block_availability_slot(
            database,
            slot_id,
        )
    except AvailabilitySlotNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )
    except InvalidAvailabilitySlotError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )


@router.post(
    "/{slot_id}/activate",
    response_model=AvailabilitySlotResponse,
)
def activate_slot(
    slot_id: int,
    database: Session = Depends(get_db),
):
    try:
        return activate_availability_slot(
            database,
            slot_id,
        )
    except AvailabilitySlotNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )
    except InvalidAvailabilitySlotError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )