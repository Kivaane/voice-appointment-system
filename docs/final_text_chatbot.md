# Final Text Chatbot

The personal appointment chatbot is hardened for text-based demo use. Voice,
telephony, RAG, deployment infrastructure, and admin features are intentionally
outside this milestone.

## Implemented safeguards

- Informational questions temporarily interrupt booking, cancellation,
  rescheduling, and availability flows without replacing the active intent.
- One-message booking extracts service, date, customer details, staff, and
  deterministic time preferences, then asks only for missing information.
- Conversation state and message history survive application restarts through
  the configured SQLAlchemy database and remain isolated by `thread_id`.
- Business dates and user-facing times use the configured `Asia/Colombo`
  timezone; technical timestamps are stored in UTC where appropriate.
- Booking and rescheduling recheck and lock mutable rows in their transaction;
  a unique appointment `slot_id` constraint is the final conflict guard.
- Read-only tool results are validated before they are shown. Database mutation
  is performed only by deterministic confirmed action handlers.
- API retries are idempotent per `(conversation_id, request_id)`.
- Browser responses retain the legacy `response` field and also provide a
  structured contract for future text or voice clients.
- Transaction, slot-option, pending-confirmation, and executing-request state
  have configurable expiry.

## Voice-readiness gate

The seven blocking text foundations are implemented and regression tested:

1. Intent interruption and recovery.
2. One-message booking.
3. Durable conversation state.
4. Asia/Colombo date and time behavior.
5. Slot-conflict protection.
6. Deterministic tool selection and output validation.
7. Confirmation and duplicate-action protection.

This means voice work may begin as a separate milestone. Voice is not included
in the current code.

## Known limitations

- SQLite ignores true row-level `FOR UPDATE` locks. The unique slot constraint
  still prevents duplicate committed bookings, while PostgreSQL provides the
  intended row-lock behavior for production-style concurrency.
- The checkpoint implementation supplements LangGraph with the project's
  existing SQLAlchemy conversation tables; it is not LangGraph's optional
  PostgreSQL checkpoint package.
- Opening hours, location, insurance, payment, and cancellation-policy facts
  remain controlled front-desk fallbacks until verified organization data is
  configured.
- The browser is text-only. No microphone, speech-to-text, text-to-speech,
  WebRTC, Twilio, or telephone integration is present.
- The suite has one existing Starlette/httpx TestClient deprecation warning.
