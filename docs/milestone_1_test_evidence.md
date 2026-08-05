# Milestone 1 Test Evidence

## Safety and intent

- `test_intent_interruption_recovery.py`
- `test_mixed_intent_safety.py`
- `test_confirmation_safety.py`
- `test_agent_controlled_fallbacks.py`
- `test_semantic_nlu_fallback.py`

## Extraction and corrections

- `test_one_message_booking.py`
- `test_milestone_one_corrections.py`
- `test_detail_extraction.py`
- `test_phone_validation.py`
- `test_precise_clarifications.py`
- `test_slot_selection_regressions.py`

## Complete appointment behavior

- `test_conversational_flow.py`
- `test_cancellation_flow.py`
- `test_reschedule_regressions.py`
- `test_chatbot_database_end_to_end.py`
- `test_slot_concurrency.py`
- `test_action_idempotency.py`

## Runtime and client contract

- `test_durable_checkpointing.py`
- `test_business_timezone.py`
- `test_conversation_expiry.py`
- `test_tool_selection_contracts.py`
- `test_structured_ai_chat.py`
- `test_browser_chat_polish.py`

All standard tests replace the semantic classifier with a no-op unless a test
explicitly supplies a fake `NLUModelResult`. Therefore the suite consumes no
live Gemini quota.

Final commands:

```bat
venv\Scripts\python.exe -m py_compile app\ai\nlu.py app\ai\agent.py
venv\Scripts\python.exe -m pytest -q
git diff --check
```
