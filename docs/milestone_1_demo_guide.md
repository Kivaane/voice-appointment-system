# Milestone 1 Demo Guide

## Start

```bat
cd C:\Users\ASUS\Desktop\voice-appointment-system
venv\Scripts\python.exe -m alembic upgrade head
venv\Scripts\python.exe scripts\reset_demo_data.py
venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

`reset_demo_data.py` deletes local demo appointments and slots before recreating
them. Run it only against disposable demo data.

Open `http://127.0.0.1:8000/chat`.

## Short CEO/TL demonstration

1. Send: `I need dental tomorrow in the morning. My name is Kivaane Anton and my number is 0774588691.`
2. Show that supplied fields are retained and no appointment exists before a
   slot and confirmation.
3. Ask: `How much is dental?` Show the concise answer and contextual resume.
4. Correct the choice: `Afternoon instead.` Show that service/date/customer are
   retained while slot options refresh.
5. Choose a slot, review the readable summary, then answer `yes`.
6. Press Send twice quickly during another request; show the disabled UI and one
   network request with a unique `request_id`.
7. Start rescheduling with the reference, ask for the cancellation policy, then
   continue and reject the reschedule to show the original remains unchanged.
8. Start cancellation, review the summary, and confirm once.
9. Select **New chat** and show that the browser creates a new `thread_id` and
   clears visible messages.
10. Finish with a handoff request: `I want to speak to a person.`

## Browser acceptance checklist

- Loading text is visible and Send/Input/New chat are disabled in flight.
- Duplicate form submissions are ignored while `isSending` is true.
- Each request carries a generated `request_id`.
- Messages automatically scroll to the latest response.
- Network failures show a friendly retry message.
- New chat changes the stored thread ID and clears the transcript.
- Confirmation summaries are readable on desktop and mobile widths.
- There are no microphone or voice controls.
