from app.ai.model import get_chat_model
from app.ai.tools import list_available_services


def main() -> None:
    model = get_chat_model()

    model_with_tools = model.bind_tools(
        [list_available_services]
    )

    response = model_with_tools.invoke(
        "What appointment services are currently available?"
    )

    print("Model response:")
    print(response.content)

    print("\nRequested tool calls:")
    print(response.tool_calls)


if __name__ == "__main__":
    main()