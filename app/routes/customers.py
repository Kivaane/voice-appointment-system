from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.customer_services import (
    CustomerAlreadyExistsError,
    CustomerNotFoundError,
    create_customer,
    deactivate_customer,
    get_customer_by_id,
    list_customers,
    update_customer,
)
from app.database import get_db
from app.schemas import (
    CustomerCreate,
    CustomerResponse,
    CustomerUpdate,
)

router = APIRouter(
    prefix="/customers",
    tags=["Customers"],
)


@router.post(
    "",
    response_model=CustomerResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_new_customer(
    customer: CustomerCreate,
    database: Session = Depends(get_db),
):
    try:
        return create_customer(
            database,
            customer,
        )
    except CustomerAlreadyExistsError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        )


@router.get(
    "",
    response_model=list[CustomerResponse],
)
def get_all_customers(
    include_inactive: bool = Query(default=False),
    database: Session = Depends(get_db),
):
    return list_customers(
        database=database,
        include_inactive=include_inactive,
    )


@router.get(
    "/{customer_id}",
    response_model=CustomerResponse,
)
def get_single_customer(
    customer_id: int,
    database: Session = Depends(get_db),
):
    try:
        return get_customer_by_id(
            database=database,
            customer_id=customer_id,
        )
    except CustomerNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )


@router.patch(
    "/{customer_id}",
    response_model=CustomerResponse,
)
def edit_customer(
    customer_id: int,
    customer: CustomerUpdate,
    database: Session = Depends(get_db),
):
    try:
        return update_customer(
            database=database,
            customer_id=customer_id,
            customer_data=customer,
        )
    except CustomerAlreadyExistsError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        )
    except CustomerNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )


@router.post(
    "/{customer_id}/deactivate",
    response_model=CustomerResponse,
)
def disable_customer(
    customer_id: int,
    database: Session = Depends(get_db),
):
    try:
        return deactivate_customer(
            database=database,
            customer_id=customer_id,
        )
    except CustomerNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )