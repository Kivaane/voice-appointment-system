from sqlalchemy import text

from app.database import engine


TABLES = [
    "ai_conversations",
    "ai_messages",
    "ai_events",
    "customers",
    "appointments",
    "availability_slots",
]


def shorten(value: object) -> object:
    if isinstance(value, str) and len(value) > 140:
        return value[:140] + "..."
    return value


def print_table(table_name: str) -> None:
    print("\n" + "=" * 80)
    print(table_name)
    print("=" * 80)

    with engine.connect() as connection:
        count = connection.execute(
            text(f"SELECT COUNT(*) FROM {table_name}")
        ).scalar_one()

        print(f"count: {count}")

        rows = connection.execute(
            text(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 10")
        ).mappings().all()

        for row in rows:
            cleaned_row = {
                key: shorten(value)
                for key, value in dict(row).items()
            }

            print(cleaned_row)


def main() -> None:
    for table_name in TABLES:
        print_table(table_name)


if __name__ == "__main__":
    main()