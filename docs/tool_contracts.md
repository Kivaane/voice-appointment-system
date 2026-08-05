# Tool and Service Contracts

| User need | Allowed operation | Validation before response or mutation |
| --- | --- | --- |
| Service list | Active service query | Service is active; safe fields only |
| Price/duration/details | Service query | Known active service; no availability or mutation call |
| Availability | Availability query | Valid IDs, matching service/staff, available status, valid datetime range |
| Booking | Availability then confirmed service-layer create | Customer/service/staff active, assignment matches, slot available, explicit pending confirmation |
| Appointment status | Appointment query | Appointment or customer identity resolves |
| Cancellation | Confirmed service-layer cancel | Appointment exists and is confirmed |
| Rescheduling | Availability then confirmed service-layer reschedule | Appointment confirmed, new slot different/available, same service, staff supports service |
| Organization information | Controlled response | No appointment mutation |

The language model may answer a general question or request a read-only tool,
but it cannot authorize booking, cancellation, or rescheduling. Those mutations
are owned by deterministic code and require a valid pending action.

Malformed tool output and exceptions return a safe message. Raw exceptions,
database internals, secrets, and private logs are not part of the chat API.

## Structured chat response

```json
{
  "thread_id": "demo-thread-001",
  "response": "I found these available slots...",
  "message": "I found these available slots...",
  "intent": "book_appointment",
  "conversation_stage": "selecting_slot",
  "requires_confirmation": false,
  "pending_action": null,
  "options": [
    {
      "id": 12,
      "label": "10:00 AM with Dr. Perera",
      "start_datetime": "2026-08-06T10:00:00",
      "end_datetime": "2026-08-06T10:30:00"
    }
  ],
  "error": null
}
```

`response` is retained for existing browser compatibility; new clients should
prefer `message` plus the structured metadata.
