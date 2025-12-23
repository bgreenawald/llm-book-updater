from typing import Any, Optional, Tuple

from loguru import logger

from src.core.constants import (
    PROMPT_PREVIEW_MAX_LENGTH,
)
from src.models.base import ProviderClient
from src.models.config import (
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
    ModelConfig,
)
from src.models.exceptions import (
    GenerationFailedError,
    LlmModelError,
    MaxRetriesExceededError,
    ResponseTruncatedError,
)
from src.models.providers import ClaudeClient, GeminiClient, OpenAIClient, OpenRouterClient
from src.models.utils import is_failed_response

__all__ = [
    "LlmModel",
    "ModelConfig",
    "ProviderClient",
    "LlmModelError",
    "GenerationFailedError",
    "MaxRetriesExceededError",
    "ResponseTruncatedError",
    "is_failed_response",
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

from src.api.provider import Provider
from src.utils.settings import settings

# Initialize module-level logger
module_logger = logger


class LlmModel:
    """
    Unified LLM client that supports multiple providers (OpenRouter, OpenAI, Gemini, Claude).
    Routes requests to the appropriate provider based on model configuration.
    """

    DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_OPENROUTER_API_ENV = "OPENROUTER_API_KEY"
    DEFAULT_OPENAI_API_ENV = "OPENAI_API_KEY"
    DEFAULT_GEMINI_API_ENV = "GEMINI_API_KEY"
    DEFAULT_CLAUDE_API_ENV = "ANTHROPIC_API_KEY"
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0  # seconds
    DEFAULT_BACKOFF_FACTOR = 2.0

    def __init__(
        self,
        model: ModelConfig,
        # OpenRouter settings
        openrouter_api_key_env: str = DEFAULT_OPENROUTER_API_ENV,
        openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        openrouter_max_retries: int = DEFAULT_MAX_RETRIES,
        openrouter_retry_delay: float = DEFAULT_RETRY_DELAY,
        openrouter_backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        # Provider-specific API key environments
        openai_api_key_env: str = DEFAULT_OPENAI_API_ENV,
        gemini_api_key_env: str = DEFAULT_GEMINI_API_ENV,
        claude_api_key_env: str = DEFAULT_CLAUDE_API_ENV,
        # Prompt logging control
        enable_prompt_logging: Optional[bool] = None,
    ) -> None:
        """
        Initialize LLM client with provider routing capabilities.

        Args:
            model: Model configuration (ModelConfig) specifying provider and model.
            openrouter_api_key_env: Environment variable for OpenRouter API key.
            openrouter_base_url: OpenRouter base URL.
            openrouter_max_retries: Maximum retry attempts for OpenRouter.
            openrouter_retry_delay: Initial retry delay for OpenRouter.
            openrouter_backoff_factor: Backoff multiplier for OpenRouter.
            openai_api_key_env: Environment variable for OpenAI API key.
            gemini_api_key_env: Environment variable for Gemini API key.
            claude_api_key_env: Environment variable for Claude (Anthropic) API key.
            enable_prompt_logging: Whether to enable prompt content logging (defaults to False).
                If None, checks LLM_ENABLE_PROMPT_LOGGING environment variable.
        """
        self.model_config = model

        # Determine prompt logging setting
        if enable_prompt_logging is None:
            self.enable_prompt_logging = settings.llm_enable_prompt_logging
        else:
            self.enable_prompt_logging = enable_prompt_logging

        # Initialize provider clients
        self._clients: dict[Provider, ProviderClient] = {}

        # Store configuration for lazy initialization
        # Explicit typing avoids mypy treating this as 'object'
        self._config: dict[str, dict[str, Any]] = {
            "openrouter": {
                "api_key_env": openrouter_api_key_env,
                "base_url": openrouter_base_url,
                "max_retries": openrouter_max_retries,
                "retry_delay": openrouter_retry_delay,
                "backoff_factor": openrouter_backoff_factor,
            },
            "openai": {"api_key_env": openai_api_key_env},
            "gemini": {"api_key_env": gemini_api_key_env},
            "claude": {"api_key_env": claude_api_key_env},
        }

        module_logger.info(
            f"Initializing LLM client: provider={self.model_config.provider.value}, model={self.model_config.model_id}"
        )

        # Validate that we have the required API key for the provider
        self._validate_api_key()

        module_logger.success(f"LLM client ready: {self.model_config.model_id}")

    def _validate_api_key(self) -> None:
        """Validate that the required API key is available for the provider."""
        provider = self.model_config.provider
        api_key = settings.get_api_key(provider.value)

        if not api_key:
            msg = f"Missing API key for provider: {provider.value}"
            module_logger.error(msg)
            raise ValueError(msg)

    def _get_client(self) -> ProviderClient:
        """Get or create the appropriate provider client."""
        provider = self.model_config.provider

        if provider not in self._clients:
            api_key = settings.get_api_key(provider.value)
            if api_key is None:
                raise ValueError(f"Missing API key for provider: {provider.value}")

            if provider == Provider.OPENROUTER:
                config = self._config["openrouter"]
                self._clients[provider] = OpenRouterClient(
                    api_key=api_key,
                    base_url=config["base_url"],
                    max_retries=config["max_retries"],
                    retry_delay=config["retry_delay"],
                    backoff_factor=config["backoff_factor"],
                )
            elif provider == Provider.OPENAI:
                self._clients[provider] = OpenAIClient(api_key=api_key)
            elif provider == Provider.GEMINI:
                self._clients[provider] = GeminiClient(api_key=api_key)
            elif provider == Provider.CLAUDE:
                self._clients[provider] = ClaudeClient(api_key=api_key)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        return self._clients[provider]

    @classmethod
    def create(
        cls,
        model: ModelConfig,
        openrouter_api_key_env: str = DEFAULT_OPENROUTER_API_ENV,
        openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        openrouter_max_retries: int = DEFAULT_MAX_RETRIES,
        openrouter_retry_delay: float = DEFAULT_RETRY_DELAY,
        openrouter_backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        openai_api_key_env: str = DEFAULT_OPENAI_API_ENV,
        gemini_api_key_env: str = DEFAULT_GEMINI_API_ENV,
        claude_api_key_env: str = DEFAULT_CLAUDE_API_ENV,
        enable_prompt_logging: Optional[bool] = None,
    ) -> "LlmModel":
        """Create a new LlmModel instance with the specified configuration."""
        return cls(
            model=model,
            openrouter_api_key_env=openrouter_api_key_env,
            openrouter_base_url=openrouter_base_url,
            openrouter_max_retries=openrouter_max_retries,
            openrouter_retry_delay=openrouter_retry_delay,
            openrouter_backoff_factor=openrouter_backoff_factor,
            openai_api_key_env=openai_api_key_env,
            gemini_api_key_env=gemini_api_key_env,
            claude_api_key_env=claude_api_key_env,
            enable_prompt_logging=enable_prompt_logging,
        )

    @property
    def model_id(self) -> str:
        """Get the model ID for backward compatibility."""
        return self.model_config.model_id

    def __str__(self) -> str:
        return f"LlmModel(provider={self.model_config.provider.value}, model_id={self.model_config.model_id})"

    def __repr__(self) -> str:
        return f"LlmModel(provider={self.model_config.provider.value}, model_id={self.model_config.model_id})"

    def _log_prompt(self, role: str, content: str) -> None:
        """
        Logs a preview of the prompt content if prompt logging is enabled.

        Args:
            role (str): The role of the prompt (e.g., "System", "User").
            content (str): The full content of the prompt.
        """
        if self.enable_prompt_logging:
            if len(content) <= PROMPT_PREVIEW_MAX_LENGTH:
                preview = content
            else:
                preview = content[:PROMPT_PREVIEW_MAX_LENGTH] + "..."
            module_logger.trace(f"{role} prompt: {preview}")

    def supports_batch(self) -> bool:
        """
        Check if the current provider supports batch processing.

        Returns:
            bool: True if batch processing is supported, False otherwise
        """
        client = self._get_client()
        return client.supports_batch

    def batch_chat_completion(
        self,
        requests: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Make batch chat completion calls using the appropriate provider.

        Args:
            requests: List of request dictionaries
            **kwargs: Additional arguments

        Returns:
            List of response dictionaries

        Raises:
            LlmModelError: When batch API calls fail
            ValueError: If provider doesn't support batch processing
        """
        client = self._get_client()

        if not client.supports_batch:
            raise ValueError(f"Provider {self.model_config.provider.value} does not support batch processing")

        # Log batch info if enabled
        if self.enable_prompt_logging:
            module_logger.trace(f"Processing batch of {len(requests)} requests")

        # Ensure model_name is a concrete string
        model_name: str = self.model_config.provider_model_name or self.model_config.model_id

        return client.batch_chat_completion(
            requests=requests,
            model_name=model_name,
            **kwargs,
        )

    def close(self) -> None:
        """
        Close all provider clients and release resources (e.g., connection pools).

        This method iterates through all initialized provider clients and calls
        their close() method if available. This is important for proper cleanup
        of HTTP sessions and connection pools, especially for OpenRouter.
        """
        for provider, client in self._clients.items():
            try:
                if hasattr(client, "close") and callable(client.close):
                    client.close()
                    module_logger.debug(f"Closed {provider.value} client")
            except Exception as e:
                # Cleanup errors are non-critical but should be logged
                module_logger.debug(f"Error closing {provider.value} client: {e!r}")

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Make a chat completion call using the appropriate provider.

        Args:
            system_prompt: System prompt for the conversation
            user_prompt: User prompt for the conversation
            **kwargs: Additional arguments

        Returns:
            Tuple of (assistant reply content, generation ID).

        Raises:
            LlmModelError: When API calls fail after max retries.
            ValueError: On empty/malformed response.
        """
        self._log_prompt(role="System", content=system_prompt)
        self._log_prompt(role="User", content=user_prompt)

        client = self._get_client()

        # Ensure model_name is a concrete string
        model_name: str = self.model_config.provider_model_name or self.model_config.model_id

        return client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=model_name,
            **kwargs,
        )
