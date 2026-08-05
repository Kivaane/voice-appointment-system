# Conversation Flows

## Booking

`intent -> service/date extraction -> availability validation -> slot choice -> customer resolution -> confirmation summary -> explicit yes -> transactional booking`

No appointment is created before a valid slot and explicit confirmation.
Temporary information questions answer first while preserving the booking
fields and stage. A rejected summary permits service, date, or time changes.

## Availability

`service + date -> validated availability query -> controlled options`

Requests such as `which date available` perform a deterministic upcoming search
and never fall through to Gemini. Invalid or failed tool output becomes a
controlled error instead of invented availability.

## Cancellation

`reference/phone -> appointment resolution -> summary -> explicit yes -> transactional cancellation`

Policy questions preserve the pending cancellation. Cancellation changes both
the appointment status and slot availability in one transaction.

## Rescheduling

`reference/phone -> appointment resolution -> new date/availability -> new slot -> summary -> explicit yes -> transactional reschedule`

The old slot is retained until commit. A failed reschedule rolls back and keeps
the original appointment. Selecting the current slot does not mutate data.

## Expiry and recovery

- Active transaction state expires after the configured transaction TTL.
- Slot options and pending confirmations have shorter independent TTLs.
- An expired confirmation cannot execute.
- Expired choices clear slot-specific fields and force a fresh availability
  query while retaining safe context such as known customer details.
- Completed or rejected flows are not reopened by the expiry handler.
