from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    StaffCreate,
    StaffResponse,
    StaffServiceAssignment,
    StaffUpdate,
)
from app.staff_services import (
    ServiceAssignmentError,
    StaffAlreadyExistsError,
    StaffNotFoundError,
    assign_service_to_staff,
    create_staff,
    deactivate_staff,
    get_staff_by_id,
    list_staff,
    remove_service_from_staff,
    update_staff,
)

router = APIRouter(
    prefix="/staff",
    tags=["Staff"],
)


@router.post(
    "",
    response_model=StaffResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_new_staff(
    staff: StaffCreate,
    database: Session = Depends(get_db),
):
    try:
        return create_staff(database, staff)
    except StaffAlreadyExistsError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        )


@router.get(
    "",
    response_model=list[StaffResponse],
)
def get_all_staff(
    include_inactive: bool = Query(False),
    database: Session = Depends(get_db),
):
    return list_staff(
        database,
        include_inactive,
    )


@router.get(
    "/{staff_id}",
    response_model=StaffResponse,
)
def get_single_staff(
    staff_id: int,
    database: Session = Depends(get_db),
):
    try:
        return get_staff_by_id(
            database,
            staff_id,
        )
    except StaffNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@router.patch(
    "/{staff_id}",
    response_model=StaffResponse,
)
def edit_staff(
    staff_id: int,
    staff: StaffUpdate,
    database: Session = Depends(get_db),
):
    try:
        return update_staff(
            database,
            staff_id,
            staff,
        )
    except StaffAlreadyExistsError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        )
    except StaffNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@router.post(
    "/{staff_id}/deactivate",
    response_model=StaffResponse,
)
def disable_staff(
    staff_id: int,
    database: Session = Depends(get_db),
):
    try:
        return deactivate_staff(
            database,
            staff_id,
        )
    except StaffNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@router.post(
    "/{staff_id}/assign-service",
    response_model=StaffResponse,
)
def assign_service(
    staff_id: int,
    assignment: StaffServiceAssignment,
    database: Session = Depends(get_db),
):
    try:
        return assign_service_to_staff(
            database,
            staff_id,
            assignment.service_id,
        )
    except (StaffNotFoundError, ServiceAssignmentError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@router.delete(
    "/{staff_id}/remove-service/{service_id}",
    response_model=StaffResponse,
)
def remove_service(
    staff_id: int,
    service_id: int,
    database: Session = Depends(get_db),
):
    try:
        return remove_service_from_staff(
            database,
            staff_id,
            service_id,
        )
    except (StaffNotFoundError, ServiceAssignmentError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )