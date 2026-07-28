from langchain_core.messages import HumanMessage, ToolMessage

from app.ai.model import get_chat_model
from app.ai.tools import list_available_services

def extract_text(content: object) -> str:
    """Extract plain text from Gemini or OpenAI response content."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []

        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
            ):
                text_parts.append(
                    str(block.get("text", ""))
                )

        return "".join(text_parts)

    return str(content)
def main() -> None:
    model = get_chat_model()

    tools = [
        list_available_services,
    ]

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    model_with_tools = model.bind_tools(tools)

    messages = [
        HumanMessage(
            content="What appointment services are currently available?"
        )
    ]

    model_response = model_with_tools.invoke(messages)
    messages.append(model_response)

    for tool_call in model_response.tool_calls:
        tool_name = tool_call["name"]
        tool_arguments = tool_call["args"]

        selected_tool = tools_by_name.get(tool_name)

        if selected_tool is None:
            raise RuntimeError(
                f"Unknown tool requested: {tool_name}"
            )

        tool_result = selected_tool.invoke(
            tool_arguments
        )

        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            )
        )

    final_response = model_with_tools.invoke(messages)

    print("\nFinal AI response:")
    print(extract_text(final_response.content))




if __name__ == "__main__":
    main()