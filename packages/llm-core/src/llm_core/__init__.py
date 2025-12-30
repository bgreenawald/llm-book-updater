"""LLM Core - Shared LLM provider integration, cost tracking, and utilities."""

__version__ = "0.1.0"

from llm_core.config import settings, Settings, BaseConfig
from llm_core.exceptions import (
    LlmModelError,
    GenerationFailedError,
    MaxRetriesExceededError,
    ResponseTruncatedError,
)
from llm_core.models import LlmModel
from llm_core.providers import (
    Provider,
    ModelConfig,
    ProviderClient,
    OpenRouterClient,
    OpenAIClient,
    GeminiClient,
    ClaudeClient,
    # Model constants
    GROK_3_MINI,
    GEMINI_FLASH,
    GEMINI_PRO,
    DEEPSEEK,
    OPENAI_04_MINI,
    CLAUDE_4_SONNET,
    GEMINI_FLASH_LITE,
    KIMI_K2,
    CLAUDE_OPUS_4_5,
    CLAUDE_SONNET_4_5,
    CLAUDE_HAIKU_4_5,
)
from llm_core.utils import is_failed_response

__all__ = [
    # Version
    "__version__",
    # Main client
    "LlmModel",
    # Config
    "settings",
    "Settings",
    "BaseConfig",
    # Providers
    "Provider",
    "ModelConfig",
    "ProviderClient",
    "OpenRouterClient",
    "OpenAIClient",
    "GeminiClient",
    "ClaudeClient",
    # Exceptions
    "LlmModelError",
    "GenerationFailedError",
    "MaxRetriesExceededError",
    "ResponseTruncatedError",
    # Utilities
    "is_failed_response",
    # Model constants
    "GROK_3_MINI",
    "GEMINI_FLASH",
    "GEMINI_PRO",
    "DEEPSEEK",
    "OPENAI_04_MINI",
    "CLAUDE_4_SONNET",
    "GEMINI_FLASH_LITE",
    "KIMI_K2",
    "CLAUDE_OPUS_4_5",
    "CLAUDE_SONNET_4_5",
    "CLAUDE_HAIKU_4_5",
]
