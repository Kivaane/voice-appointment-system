# Testing

## Commands

```bat
venv\Scripts\python.exe -m py_compile app\ai\nlu.py app\ai\agent.py app\appointment_services.py app\ai_persistence.py
venv\Scripts\python.exe -m pytest -q
git --no-pager diff --check
```

Run migrations on a local database with:

```bat
venv\Scripts\python.exe -m alembic upgrade head
```

## Permanent coverage map

- Booking and change/rejection flows: `test_conversational_flow.py`,
  `test_one_message_booking.py`, `test_slot_selection_regressions.py`.
- Cancellation identification, confirmation, rejection, and already-cancelled
  behavior: `test_cancellation_flow.py`, `test_confirmation_safety.py`.
- Rescheduling and regression paths: `test_reschedule_regressions.py`,
  `test_confirmation_safety.py`.
- Information interruption and conversation control:
  `test_intent_interruption_recovery.py`,
  `test_agent_controlled_fallbacks.py`.
- Durable state and thread isolation: `test_durable_checkpointing.py`.
- Asia/Colombo boundaries: `test_business_timezone.py`.
- Conflict/rollback behavior: `test_slot_concurrency.py`.
- Tool choice and malformed results: `test_tool_selection_contracts.py`.
- Request/action duplication: `test_action_idempotency.py`.
- Structured browser contract: `test_structured_ai_chat.py` and
  `test_ai_chat.py`.
- Expiry and stale choice recovery: `test_conversation_expiry.py`.
- Real booking/reschedule/cancel database outcomes:
  `test_chatbot_database_end_to_end.py`.

The database end-to-end test verifies one appointment row is created, the slot
is booked, a repeated completed confirmation makes no duplicate, rescheduling
releases the old slot and reserves the new one, and cancellation persists its
status/reason while releasing the new slot.
