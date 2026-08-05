from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app


TEST_DATABASE_URL = "sqlite://"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={
        "check_same_thread": False,
    },
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(
    bind=test_engine,
    autoflush=False,
    autocommit=False,
)


def override_get_db() -> Generator[Session, None, None]:
    """Provide a test database session to FastAPI."""

    database = TestingSessionLocal()

    try:
        yield database
    finally:
        database.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def reset_test_database() -> Generator[None, None, None]:
    """Create a clean database before every test."""

    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    yield

    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(autouse=True)
def prevent_live_semantic_nlu(monkeypatch) -> None:
    """Standard tests must never consume live model quota."""

    monkeypatch.setattr(
        "app.ai.agent.classify_unknown_message",
        lambda user_message: None,
    )


@pytest.fixture(autouse=True)
def isolate_internal_database_sessions(monkeypatch) -> None:
    """Keep agent and tool persistence out of the developer demo database."""

    monkeypatch.setattr(
        "app.ai.agent.SessionLocal",
        TestingSessionLocal,
    )
    monkeypatch.setattr(
        "app.ai.tools.SessionLocal",
        TestingSessionLocal,
    )


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provide a FastAPI test client."""

    with TestClient(app) as test_client:
        yield test_client
