# Demo Guide

## Prepare the personal demo

From `C:\Users\ASUS\Desktop\voice-appointment-system`:

```bat
venv\Scripts\python.exe -m alembic upgrade head
venv\Scripts\python.exe scripts\reset_demo_data.py
venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The reset command deliberately deletes existing local appointments and
availability slots before recreating predictable demo data. Do not run it
against data you need to retain.

Open `http://127.0.0.1:8000/chat` and keep the returned thread ID for a single
conversation. Use a new thread ID to demonstrate isolation.

## Suggested demo script

1. `I need dental tomorrow in the morning. My name is Kivaane Anton and my number is 0774588691.`
2. Select a displayed slot.
3. Ask `How much is dental?` and show that the booking remains active.
4. Continue and reply `yes` only after the confirmation summary.
5. Repeat the same API request with the same `request_id` and show that no
   second appointment is created.
6. Start a reschedule using the returned reference, select another slot, ask
   `Where are you located?`, then continue and confirm.
7. Start cancellation, ask for the cancellation policy, then return and
   confirm cancellation.
8. Ask `which dates are available` to show controlled upcoming availability.

## Browser smoke-test checklist

- `/health` returns `{"status":"healthy"}`.
- `/chat` loads without an endless loading indicator.
- The first message returns a stable `thread_id`.
- Booking shows structured slot options and does not mutate before `yes`.
- An informational interruption preserves the active transaction.
- `never mind`, `stop`, and human handoff stop the active flow.
- Rescheduling keeps the original appointment until confirmed.
- Cancellation releases the slot only after confirmation.
- Reusing one `request_id` returns the stored response once.
- Provider or availability failures show controlled text and clear the loading
  state.
- Starting the server again and reusing the same `thread_id` continues the
  durable conversation.
