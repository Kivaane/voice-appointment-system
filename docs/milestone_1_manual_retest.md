# Milestone 1 manual retest matrix

Date: 2026-08-05

Project: Personal AI Voice Appointment System
Branch: `feat/smart-nlu-router`

This matrix records the browser failures supplied during the Milestone 1 review
and the deterministic regression coverage added for each one. Automated tests
use the isolated test database and stub semantic NLU, so they consume no live
Gemini quota. A final browser click-through should still be performed after the
local application is restarted with the reviewed working tree.

| ID | Thread | User message(s) | Expected and actual automated behaviour | Result | Regression coverage |
|---|---|---|---|---|---|
| M1-01 | Fresh | `hello`, `how are you`, `what can you do` | Controlled greeting/capability response; no transaction starts. | PASS | `tests/test_agent_controlled_fallbacks.py`, `tests/test_browser_chat_polish.py` |
| M1-02 | Fresh | `talk to a human`, `human please`, `staff member`, `connect me to someone` | Immediate concise handoff; semantic NLU and chat model are bypassed. | PASS | `tests/test_manual_bugfix_regressions.py` |
| M1-03 | Fresh | `what services do you have?` | Verified active service list only. | PASS | `tests/test_structured_ai_chat.py`, `tests/test_tool_selection_contracts.py` |
| M1-04 | Fresh | `how much is dental?`, `duration of physiotherapy` | Direct verified answer; no booking/date question is appended. | PASS | `tests/test_manual_bugfix_regressions.py`, `tests/test_intent_interruption_recovery.py` |
| M1-05 | Fresh | opening hours, location, insurance, payment, cancellation policy | Says verified organisation data is unavailable and directs the user to the front desk. | PASS | `tests/test_milestone_one_corrections.py`, `tests/test_precise_clarifications.py` |
| M1-06 | Continuing booking | Ask price/duration while choosing a slot | Answers first, then resumes exactly one genuine pending booking question. | PASS | `tests/test_manual_bugfix_regressions.py`, `tests/test_intent_interruption_recovery.py` |
| M1-07 | Continuing reschedule | `I don't want my appointment anymore` | Switches to cancellation, clears pending reschedule choices, performs no mutation. | PASS | `tests/test_manual_bugfix_regressions.py` |
| M1-08 | Continuing booking | `I can't come for my appointment` | Switches to cancellation without carrying unsafe booking choices. | PASS | `tests/test_manual_bugfix_regressions.py` |
| M1-09 | Continuing cancellation | `I need to reschedule my appointment` | Switches to rescheduling; no action occurs before confirmation. | PASS | `tests/test_manual_bugfix_regressions.py` |
| M1-10 | Continuing reschedule | `What time is my appointment?` | Temporarily enters appointment-status lookup and records the paused reschedule intent for safe resumption. | PASS | `tests/test_manual_bugfix_regressions.py` |
| M1-11 | Fresh | `is dental available tomorrorw` | Controlled typo normalization resolves `tomorrow` and preserves the original logged message. | PASS | `tests/test_manual_bugfix_regressions.py` |
| M1-12 | Fresh | `do u offer phydyotherapy` | Resolves Physiotherapy from the controlled typo map. | PASS | `tests/test_manual_bugfix_regressions.py` |
| M1-13 | Fresh | `do u do surger` | Corrects to `surgery`, then reports it as unsupported rather than inventing a service. | PASS | `tests/test_manual_bugfix_regressions.py` |
| M1-14 | Continuing availability | dental tomorrow → physio next Monday → dental times | Each explicit service replaces the previous service and clears dependent staff/slot choices while preserving the new date in the same message. | PASS | `tests/test_manual_bugfix_regressions.py` |
| M1-15 | Continuing availability | `when is available` | Clears stale date/results and asks the precise missing date question. | PASS | `tests/test_manual_bugfix_regressions.py` |
| M1-16 | Fresh | Monday / next Monday / this Friday from 2026-08-05 | Resolves to 2026-08-10 / 2026-08-10 / 2026-08-07. | PASS | `tests/test_manual_bugfix_regressions.py`, `tests/test_detail_extraction.py` |
| M1-17 | Fresh | `on the 6th`, long-message date, conflicting dates | Chooses the next valid ordinal day; long-message extraction works; conflicts require clarification. | PASS | `tests/test_manual_bugfix_regressions.py`, `tests/test_precise_clarifications.py` |
| M1-18 | Continuing availability | morning / afternoon / evening | Applies documented non-overlapping time ranges. | PASS | `tests/test_manual_bugfix_regressions.py` |
| M1-19 | Continuing availability | `after 2 pm`, `after 2 p.m.`, `after two pm`, `before noon` | Normalizes safe clock variants and filters at the documented boundaries without mutating source slots. | PASS | `tests/test_manual_bugfix_regressions.py` |
| M1-20 | Continuing availability | `earliest available`, `latest available` | Returns only the first or last matching slot. | PASS | `tests/test_manual_bugfix_regressions.py` |
| M1-21 | Service/slot choice | `first`, `1st`, `second`, `2nd`, `2` | Standalone safe ordinals select known options; phone/reference/date text is not treated as an option. | PASS | `tests/test_manual_bugfix_regressions.py`, `tests/test_slot_selection_regressions.py` |
| M1-22 | Fresh | Standard multi-turn booking | Checks real availability, gathers missing details, shows summary, and requires explicit confirmation before mutation. | PASS | `tests/test_conversational_flow.py`, `tests/test_chatbot_database_end_to_end.py` |
| M1-23 | Fresh | Four supplied one-message booking examples | Extracts intent, service, date, time preference, staff, name, and phone when present; asks only for genuinely missing data. | PASS | `tests/test_manual_bugfix_regressions.py`, `tests/test_one_message_booking.py` |
| M1-24 | Continuing after completion/rejection/cancel/reschedule | `book another appointment` | Clears every transaction-specific field; intentionally retained verified customer data stays in the merged conversation state. | PASS | `tests/test_manual_bugfix_regressions.py`, `tests/test_conversational_flow.py` |
| M1-25 | Continuing after success | repeat `yes` after booking/cancel/reschedule | Returns concise “already …” response with reference and performs no second mutation. | PASS | `tests/test_manual_bugfix_regressions.py`, `tests/test_action_idempotency.py` |
| M1-26 | Continuing after no slots | `which date available` and supported variants | Searches upcoming verified availability and always returns a controlled `next_question`; no Gemini fallback or indefinite loading. | PASS | `tests/test_upcoming_availability.py`, `tests/test_system_prompt_contract.py` |
| M1-27 | Fresh/reused durable thread | expired transaction followed by a new booking request | Clears unsafe stale data but still processes the current fresh request. | PASS | `tests/test_conversation_expiry.py`, `tests/test_simple_agent.py` |
| M1-28 | Separate tests | reused thread identifier | Agent/tools use the isolated test session, preventing production/demo checkpoint leakage between tests. | PASS | `tests/conftest.py`, `tests/test_durable_checkpointing.py` |
| M1-29 | Any | prompt/model fallback | Uses the imported appointment prompt, verified-data and mutation safety rules; controlled `next_question` responses bypass the model. | PASS | `tests/test_system_prompt_contract.py`, `tests/test_semantic_nlu_fallback.py` |

## Browser follow-up

Restart the local API/UI before browser verification so the process loads the
new Python modules. Repeat the messages above in fresh and continuing chats.
The browser row is complete only when the visible response matches the tested
controlled response and the loading indicator clears. Do not use a live model
to validate deterministic handoff, availability-list, confirmation, or safety
paths; those paths must already contain `next_question` before `call_model`.
