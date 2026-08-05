import inspect

from langchain_core.messages import HumanMessage, SystemMessage

from app.ai import agent
from app.ai.prompts import APPOINTMENT_SYSTEM_PROMPT


def test_prompt_is_exported_and_not_defined_inline_in_agent() -> None:
    source = inspect.getsource(agent)

    assert len(APPOINTMENT_SYSTEM_PROMPT) > 1000
    assert "SYSTEM_PROMPT =" not in source
    assert "APPOINTMENT_SYSTEM_PROMPT" in source


def test_prompt_contains_current_capabilities_and_safety_contract() -> None:
    prompt = APPOINTMENT_SYSTEM_PROMPT.lower()

    for phrase in (
        "book, view, cancel, and reschedule",
        "verified service prices and durations",
        "never invent",
        "never perform, authorize",
        "general information question",
        "human-handoff",
        "front desk",
        "one main follow-up question",
    ):
        assert phrase in prompt

    assert "rag" not in prompt
    assert "voice" not in prompt
    assert "reminders are available" not in prompt


def test_call_model_uses_imported_prompt(monkeypatch) -> None:
    captured = []

    class FakeModel:
        def bind_tools(self, _tools):
            return self

        def invoke(self, messages):
            captured.extend(messages)
            return agent.LangChainAIMessage(content="controlled fake")

    monkeypatch.setattr(agent, "get_chat_model", lambda: FakeModel())
    result = agent.call_model(
        {"messages": [HumanMessage(content="unclassified question")]}
    )

    assert result["messages"][0].content == "controlled fake"
    assert isinstance(captured[0], SystemMessage)
    assert captured[0].content == APPOINTMENT_SYSTEM_PROMPT


def test_next_question_bypasses_model(monkeypatch) -> None:
    monkeypatch.setattr(
        agent,
        "get_chat_model",
        lambda: (_ for _ in ()).throw(AssertionError("model must not be called")),
    )

    result = agent.call_model(
        {"messages": [], "next_question": "Deterministic response."}
    )

    assert result["messages"][0].content == "Deterministic response."
