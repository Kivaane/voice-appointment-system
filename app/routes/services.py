from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    ServiceCreate,
    ServiceResponse,
    ServiceUpdate,
)
from app.services import (
    ServiceAlreadyExistsError,
    ServiceNotFoundError,
    create_service,
    deactivate_service,
    get_service_by_id,
    list_services,
    update_service,
)

router = APIRouter(
    prefix="/services",
    tags=["Services"],
)


@router.post(
    "",
    response_model=ServiceResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_new_service(
    service: ServiceCreate,
    database: Session = Depends(get_db),
):
    try:
        return create_service(database, service)
    except ServiceAlreadyExistsError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        )


@router.get(
    "",
    response_model=list[ServiceResponse],
)
def get_all_services(
    include_inactive: bool = Query(False),
    database: Session = Depends(get_db),
):
    return list_services(
        database,
        include_inactive,
    )


@router.get(
    "/{service_id}",
    response_model=ServiceResponse,
)
def get_single_service(
    service_id: int,
    database: Session = Depends(get_db),
):
    try:
        return get_service_by_id(
            database,
            service_id,
        )
    except ServiceNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@router.patch(
    "/{service_id}",
    response_model=ServiceResponse,
)
def edit_service(
    service_id: int,
    service: ServiceUpdate,
    database: Session = Depends(get_db),
):
    try:
        return update_service(
            database,
            service_id,
            service,
        )
    except ServiceAlreadyExistsError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        )
    except ServiceNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@router.post(
    "/{service_id}/deactivate",
    response_model=ServiceResponse,
)
def disable_service(
    service_id: int,
    database: Session = Depends(get_db),
):
    try:
        return deactivate_service(
            database,
            service_id,
        )
    except ServiceNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )