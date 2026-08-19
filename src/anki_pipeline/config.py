"""Configuration management for the Anki Pipeline.

All sensitive data is loaded from environment variables to ensure
no credentials are exposed in the codebase (security-first approach).
"""

import os

from dotenv import load_dotenv

from .errors import ConfigurationError

load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # LLM Server Configuration (Ollama with Intel GPU via Vulkan)
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "qwen2.5-coder:3b")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "placeholder")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")

    # Siyuan Notes Configuration
    SIYUAN_API_URL: str = os.getenv(
        "SIYUAN_API_URL", "http://127.0.0.1:6806/api/block/getBlockKramdown"
    )
    SIYUAN_API_TOKEN: str = os.getenv("SIYUAN_API_TOKEN", "")
    TARGET_BLOCK_ID: str = os.getenv("TARGET_BLOCK_ID", "")

    # Anki Configuration
    ANKI_CONNECT_URL: str = os.getenv("ANKI_CONNECT_URL", "http://localhost:8765")
    ANKI_DECK_NAME: str = os.getenv("ANKI_DECK_NAME", "Default")

    def validate(self) -> list[str]:
        """Return human-readable configuration validation errors."""
        errors = []
        if not self.TARGET_BLOCK_ID:
            errors.append("TARGET_BLOCK_ID is required")
        return errors

    def require_valid(self) -> None:
        """Raise a typed configuration error when required settings are missing."""
        errors = self.validate()
        if errors:
            raise ConfigurationError("; ".join(errors))


config = Config()
