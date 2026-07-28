from app.ai.model import AIConfigurationError, get_chat_model


def main() -> None:
    try:
        model = get_chat_model()

        response = model.invoke(
            "Reply with exactly: AI connection successful"
        )

        if isinstance(response.content, str):
            print(response.content)
        else:
            text_parts = [
                block.get("text", "")
                for block in response.content
                if isinstance(block, dict) and block.get("type") == "text"
            ]

        print("".join(text_parts))

    except AIConfigurationError as error:
        print(f"Configuration error: {error}")

    except Exception as error:
        print(
            f"AI request failed: "
            f"{type(error).__name__}: {error}"
        )


if __name__ == "__main__":
    main()