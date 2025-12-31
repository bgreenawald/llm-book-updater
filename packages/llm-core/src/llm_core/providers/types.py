"""Provider types and configuration for LLM Core."""

from enum import Enum
from typing import Optional

from pydantic import model_validator

from llm_core.config.pydantic_config import BaseConfig


class Provider(Enum):
    """Enumeration of supported LLM providers."""

    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"


class ModelConfig(BaseConfig):
    """Configuration for a model, including provider and model identifier."""

    provider: Provider
    model_id: str
    # Provider-specific model name (for direct SDK calls)
    provider_model_name: Optional[str] = None

    @model_validator(mode="after")
    def _set_provider_model_name(self) -> "ModelConfig":
        """Set provider_model_name if not specified."""
        if self.provider_model_name is None:
            if self.provider in (Provider.OPENAI, Provider.GEMINI):
                self.provider_model_name = self.model_id.split("/", 1)[1] if "/" in self.model_id else self.model_id
            else:
                self.provider_model_name = self.model_id
        return self


# Model constants with provider information
GROK_3_MINI = ModelConfig(provider=Provider.OPENROUTER, model_id="x-ai/grok-3-mini")
GEMINI_FLASH = ModelConfig(
    provider=Provider.GEMINI,
    model_id="google/gemini-2.5-flash",
    provider_model_name="gemini-2.5-flash",
)
GEMINI_PRO = ModelConfig(
    provider=Provider.GEMINI,
    model_id="google/gemini-2.5-pro",
    provider_model_name="gemini-2.5-pro",
)
DEEPSEEK = ModelConfig(provider=Provider.OPENROUTER, model_id="deepseek/deepseek-r1-0528")
OPENAI_04_MINI = ModelConfig(
    provider=Provider.OPENAI,
    model_id="openai/o4-mini-high",
    provider_model_name="o4-mini",
)
CLAUDE_4_SONNET = ModelConfig(provider=Provider.OPENROUTER, model_id="anthropic/claude-sonnet-4")
GEMINI_FLASH_LITE = ModelConfig(
    provider=Provider.GEMINI,
    model_id="google/gemini-2.5-flash-lite-preview-06-17",
    provider_model_name="gemini-2.5-flash-lite",
)
KIMI_K2 = ModelConfig(provider=Provider.OPENROUTER, model_id="moonshotai/kimi-k2:free")
# Claude API models (direct Anthropic API)
CLAUDE_OPUS_4_5 = ModelConfig(provider=Provider.CLAUDE, model_id="claude-opus-4-5")
CLAUDE_SONNET_4_5 = ModelConfig(provider=Provider.CLAUDE, model_id="claude-sonnet-4-5")
CLAUDE_HAIKU_4_5 = ModelConfig(provider=Provider.CLAUDE, model_id="claude-haiku-4-5")
