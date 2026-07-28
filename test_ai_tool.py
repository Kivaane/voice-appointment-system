from app.ai.tools import list_available_services


def main() -> None:
    result = list_available_services.invoke({})

    print(result)


if __name__ == "__main__":
    main()