from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from app.ai.agent import call_model


def test_returns_next_question_without_calling_model() -> None:
    with patch(
        "app.ai.agent.get_chat_model"
    ) as mocked_get_chat_model:
        result = call_model(
            {
                "messages": [
                    HumanMessage(
                        content="I need an appointment."
                    ),
                ],
                "next_question": (
                    "Which service would you like to book?"
                ),
            }
        )

    mocked_get_chat_model.assert_not_called()

    response = result["messages"][0]

    assert isinstance(response, AIMessage)
    assert (
        response.content
        == "Which service would you like to book?"
    )


def test_calls_model_when_no_next_question_exists() -> None:
    mocked_model_response = AIMessage(
        content="How may I assist you?"
    )

    with patch(
        "app.ai.agent.get_chat_model"
    ) as mocked_get_chat_model:
        mocked_model = mocked_get_chat_model.return_value
        mocked_model.bind_tools.return_value = mocked_model
        mocked_model.invoke.return_value = mocked_model_response

        result = call_model(
            {
                "messages": [
                    HumanMessage(content="Hello"),
                ],
                "next_question": None,
            }
        )

    mocked_model.invoke.assert_called_once()
    assert result["messages"][0] == mocked_model_response