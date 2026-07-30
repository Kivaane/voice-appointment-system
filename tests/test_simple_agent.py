from langchain_core.messages import AIMessage

from app.ai import agent


class FakeChatModel:
    """Small fake model used to test the graph without an API call."""

    def invoke(self, messages: list[object]) -> AIMessage:
        assert len(messages) == 2

        return AIMessage(
            content="What type of appointment would you like to book?"
        )


def test_simple_appointment_agent(monkeypatch) -> None:
    monkeypatch.setattr(
        agent,
        "get_chat_model",
        lambda: FakeChatModel(),
    )

    response = agent.run_appointment_agent(
        "I need an appointment."
    )

    assert response == (
        "What type of appointment would you like to book?"
    )

def test_extracts_text_from_structured_model_content(
    monkeypatch,
) -> None:
    class StructuredFakeModel:
        def invoke(self, messages: list[object]) -> AIMessage:
            return AIMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Please provide your preferred service.",
                    }
                ]
            )

    monkeypatch.setattr(
        agent,
        "get_chat_model",
        lambda: StructuredFakeModel(),
    )

    response = agent.run_appointment_agent(
        "I want an appointment."
    )

    assert response == "Please provide your preferred service."