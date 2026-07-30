from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = "AI Voice Appointment System"
    app_env: str = "development"
    database_url: str = "sqlite:///./appointments.db"
    currency: str = "LKR"

    ai_provider: Literal["gemini", "openai"] = "gemini"

    google_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"

    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini"



    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()