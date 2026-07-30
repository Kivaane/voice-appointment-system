from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from app.ai.agent import run_appointment_agent
from app.schemas import AIChatRequest, AIChatResponse


router = APIRouter(
    prefix="/ai",
    tags=["AI Appointment Agent"],
)


@router.post(
    "/chat",
    response_model=AIChatResponse,
    status_code=status.HTTP_200_OK,
)
def chat_with_appointment_agent(
    chat_request: AIChatRequest,
) -> AIChatResponse:
    """Send one message to the stateful appointment agent."""

    thread_id = chat_request.thread_id or str(uuid4())
    request_id = str(uuid4())

    try:
        assistant_response = run_appointment_agent(
            user_message=chat_request.message,
            thread_id=thread_id,
            request_id=request_id,
        )
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The appointment agent could not process the request.",
        ) from error

    return AIChatResponse(
        thread_id=thread_id,
        response=assistant_response,
    )