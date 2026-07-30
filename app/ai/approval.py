from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class BookingApprovalState(TypedDict):
    """State used while requesting approval for a booking action."""

    booking_details: dict[str, str]
    approval_status: str | None


def request_booking_approval(
    state: BookingApprovalState,
) -> BookingApprovalState:
    """Pause the graph until a human approves or rejects the booking."""

    approved = interrupt(
        {
            "action": "book_appointment",
            "message": "Approve this appointment booking?",
            "booking_details": state["booking_details"],
        }
    )

    return {
        "booking_details": state["booking_details"],
        "approval_status": (
            "approved"
            if approved is True
            else "rejected"
        ),
    }


def build_booking_approval_graph():
    """Build the booking approval graph."""

    graph_builder = StateGraph(BookingApprovalState)

    graph_builder.add_node(
        "request_booking_approval",
        request_booking_approval,
    )

    graph_builder.add_edge(
        START,
        "request_booking_approval",
    )

    graph_builder.add_edge(
        "request_booking_approval",
        END,
    )

    return graph_builder.compile(
        checkpointer=InMemorySaver(),
    )


booking_approval_graph = build_booking_approval_graph()


def start_booking_approval(
    thread_id: str,
    booking_details: dict[str, str],
) -> dict:
    """Start an approval request and pause at the interrupt."""

    return booking_approval_graph.invoke(
        {
            "booking_details": booking_details,
            "approval_status": None,
        },
        config={
            "configurable": {
                "thread_id": thread_id,
            }
        },
    )


def resume_booking_approval(
    thread_id: str,
    approved: bool,
) -> dict:
    """Resume the paused graph with the human decision."""

    return booking_approval_graph.invoke(
        Command(resume=approved),
        config={
            "configurable": {
                "thread_id": thread_id,
            }
        },
    )