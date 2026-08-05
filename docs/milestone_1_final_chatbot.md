# Milestone 1 Final Text Chatbot

Milestone 1 delivers a browser-based text appointment receptionist. It uses
deterministic routing first, validated semantic classification only for unknown
low-risk paraphrases, real service/availability data, and deterministic
confirmation-gated mutations.

## Routing priority

1. Expiry and safety state recovery.
2. Human handoff or explicit stop.
3. Pending yes/no confirmation and corrections.
4. Conflicting mutation clarification.
5. Deterministic NLU.
6. Validated semantic classification for unknown low-risk language only.
7. Entity extraction and database-backed resolution.
8. Read-only availability/service operations.
9. Confirmed deterministic mutation.
10. Controlled response, with general model fallback only when safe.

## Conversation quality

- Pricing, duration, policy, and organization questions preserve active work.
- Corrections invalidate only dependent state and always invalidate an old
  confirmation summary.
- Two requested mutations are ordered through clarification and never execute
  together.
- Responses answer first and contain at most one primary continuation question.
- Failed clarification offers human handoff on the third unsuccessful attempt.
- Completed actions do not repeat on another confirmation.

## Semantic classifier boundary

The semantic classifier returns the Pydantic `NLUModelResult` schema. It can
classify and extract text hints, but cannot query private appointment data,
select database records, authorize mutations, or claim success. It is skipped
for known messages, pending confirmations, appointment references, phone
numbers, numeric choices, and slot selections.

No voice functionality is part of Milestone 1.
