"""Application settings using Pydantic Settings."""

import os

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openrouter_api_key: SecretStr | None = Field(None, alias="OPENROUTER_API_KEY")
    openai_api_key: SecretStr | None = Field(None, alias="OPENAI_API_KEY")
    gemini_api_key: SecretStr | None = Field(None, alias="GEMINI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(None, alias="ANTHROPIC_API_KEY")

    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openai_base_url: str = "https://api.openai.com/v1"

    debug: bool = False
    log_level: str = "INFO"
    llm_enable_prompt_logging: bool = Field(False, alias="LLM_ENABLE_PROMPT_LOGGING")

    def get_api_key(self, provider: str) -> str | None:
        """Get the API key for a provider name.

        Args:
            provider: Provider name (openrouter, openai, gemini, anthropic, or claude)

        Returns:
            API key string if found, None otherwise
        """
        provider_key = provider.lower()
        key_map = {
            "openrouter": ("OPENROUTER_API_KEY", self.openrouter_api_key),
            "openai": ("OPENAI_API_KEY", self.openai_api_key),
            "gemini": ("GEMINI_API_KEY", self.gemini_api_key),
            "anthropic": ("ANTHROPIC_API_KEY", self.anthropic_api_key),
            "claude": ("ANTHROPIC_API_KEY", self.anthropic_api_key),
        }
        env_name, secret_value = key_map.get(provider_key, (None, None))
        if secret_value:
            return secret_value.get_secret_value()
        if env_name:
            return os.getenv(env_name)
        return None

    def get_env(self, name: str, default: str | None = None) -> str | None:
        """Get a raw environment variable, for custom overrides."""
        return os.getenv(name, default)


settings: Settings = Settings()  # type: ignore[call-arg]
