"""LLM Core - Shared LLM provider integration, cost tracking, and utilities."""

__version__ = "0.1.0"

from llm_core.config import BaseConfig, Settings, settings
from llm_core.exceptions import (
    APIError,
    AuthenticationError,
    GenerationFailedError,
    LlmModelError,
    MaxRetriesExceededError,
    RateLimitError,
    ResponseTruncatedError,
)
from llm_core.models import LlmModel
from llm_core.providers import (
    # Model constants
    CLAUDE_4_SONNET,
    CLAUDE_HAIKU_4_5,
    CLAUDE_OPUS_4_5,
    CLAUDE_SONNET_4_5,
    DEEPSEEK,
    GEMINI_FLASH,
    GEMINI_FLASH_LITE,
    GEMINI_PRO,
    GROK_3_MINI,
    KIMI_K2,
    OPENAI_04_MINI,
    # Async clients
    AsyncOpenRouterClient,
    # Base classes
    AsyncProviderClient,
    # Sync clients
    ClaudeClient,
    GeminiClient,
    # Types
    ModelConfig,
    OpenAIClient,
    OpenRouterClient,
    Provider,
    ProviderClient,
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
    # Base classes
    "ProviderClient",
    "AsyncProviderClient",
    # Sync providers
    "Provider",
    "ModelConfig",
    "OpenRouterClient",
    "OpenAIClient",
    "GeminiClient",
    "ClaudeClient",
    # Async providers
    "AsyncOpenRouterClient",
    # Exceptions
    "LlmModelError",
    "GenerationFailedError",
    "MaxRetriesExceededError",
    "ResponseTruncatedError",
    "RateLimitError",
    "APIError",
    "AuthenticationError",
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
