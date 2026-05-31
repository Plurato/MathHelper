"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings for MathCoach-Agent."""

    openrouter_api_key: str
    openrouter_base_url: str
    default_model: str
    default_temperature: float

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables."""
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            default_model=os.getenv("MATHCOACH_DEFAULT_MODEL", "openai/gpt-4o-mini"),
            default_temperature=float(os.getenv("MATHCOACH_DEFAULT_TEMPERATURE", "0.2")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings.from_env()
