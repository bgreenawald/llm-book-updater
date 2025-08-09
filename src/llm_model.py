import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

import requests
from dotenv import load_dotenv

from src.logging_config import setup_logging

# Load environment variables from .env to ensure API keys are available
load_dotenv(override=True)

# Initialize module-level logger
module_logger = setup_logging(log_name="llm_model")


class Provider(Enum):
    """Enumeration of supported LLM providers."""

    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class ModelConfig:
    """Configuration for a model, including provider and model identifier."""

    provider: Provider
    model_id: str
    # Provider-specific model name (for direct SDK calls)
    provider_model_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Set provider_model_name if not specified."""
        if self.provider_model_name is None:
            if self.provider == Provider.OPENAI:
                # Extract OpenAI model name from OpenRouter format
                if "/" in self.model_id:
                    self.provider_model_name = self.model_id.split("/", 1)[1]
                else:
                    self.provider_model_name = self.model_id
            elif self.provider == Provider.GEMINI:
                # Extract Gemini model name from OpenRouter format
                if "/" in self.model_id:
                    self.provider_model_name = self.model_id.split("/", 1)[1]
                else:
                    self.provider_model_name = self.model_id
            else:
                # For OpenRouter, use the full model_id
                self.provider_model_name = self.model_id


# Model constants with provider information
GROK_3_MINI = ModelConfig(Provider.OPENROUTER, "x-ai/grok-3-mini")
GEMINI_FLASH = ModelConfig(Provider.GEMINI, "google/gemini-2.5-flash", "gemini-2.5-flash")
GEMINI_PRO = ModelConfig(Provider.GEMINI, "google/gemini-2.5-pro", "gemini-2.5-pro")
DEEPSEEK = ModelConfig(Provider.OPENROUTER, "deepseek/deepseek-r1-0528")
OPENAI_04_MINI = ModelConfig(Provider.OPENAI, "openai/o4-mini-high", "o4-mini")
CLAUDE_4_SONNET = ModelConfig(Provider.OPENROUTER, "anthropic/claude-sonnet-4")
GEMINI_FLASH_LITE = ModelConfig(Provider.GEMINI, "google/gemini-2.5-flash-lite-preview-06-17", "gemini-2.5-flash-lite")
KIMI_K2 = ModelConfig(Provider.OPENROUTER, "moonshotai/kimi-k2:free")


class LlmModelError(Exception):
    """Custom exception for LLM model errors."""

    pass


class ProviderClient(ABC):
    """Abstract base class for provider-specific LLM clients."""

    @abstractmethod
    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        temperature: float,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Make a chat completion call.

        Returns:
            Tuple of (response_content, generation_id)
        """
        pass


class OpenRouterClient(ProviderClient):
    """Client for OpenRouter API calls."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry."""
        if isinstance(error, requests.exceptions.RequestException):
            if hasattr(error, "response") and error.response is not None:
                status_code = error.response.status_code
                return status_code >= 500 or status_code == 429
            return True
        return False

    def _make_api_call(self, headers: dict, data: dict) -> dict:
        """Makes a single API call to OpenRouter with retry logic."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url=f"{self.base_url}/chat/completions",
                    headers=headers,
                    data=json.dumps(obj=data),
                    timeout=30,
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries and self._should_retry(error=e):
                    delay = self.retry_delay * (self.backoff_factor**attempt)
                    module_logger.warning(
                        f"API call failed (attempt {attempt + 1}/"
                        f"{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    break
            except json.JSONDecodeError as e:
                last_error = e
                module_logger.error(f"Failed to parse response JSON: {e}")
                break
            except Exception as e:
                last_error = e
                module_logger.error(f"Unexpected error during API call: {e}")
                break

        error_msg = f"API call failed after {self.max_retries + 1} attempts"
        if last_error:
            error_msg += f": {last_error}"

        module_logger.error(error_msg)
        raise LlmModelError(error_msg)

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        temperature: float,
        **kwargs,
    ) -> Tuple[str, str]:
        """Make a chat completion call using OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # GPT-5 models (and variants) currently do not accept custom temperature
        # values. Detect and omit temperature to avoid 400 errors from upstream.
        provider_model = model_name.split("/", 1)[1] if "/" in model_name else model_name
        is_gpt5_series_model = provider_model.lower().startswith("gpt-5")

        # Remove unsupported parameters for OpenRouter chat completions
        # (e.g., a "reasoning" dict intended for other APIs)
        kwargs.pop("reasoning", None)

        data: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        }
        if not is_gpt5_series_model:
            data["temperature"] = temperature

        resp_data = self._make_api_call(headers=headers, data=data)

        choices = resp_data.get("choices", [])
        if not choices or not choices[0].get("message", {}).get("content"):
            raise ValueError(f"Empty or malformed response: {resp_data}")

        content = choices[0]["message"]["content"]
        finish_reason = choices[0].get("finish_reason", "unknown")

        if finish_reason == "length":
            module_logger.warning("Response truncated: consider increasing max_tokens or reviewing model limits")

        generation_id = resp_data.get("id", "unknown")
        return content, generation_id


class OpenAIClient(ProviderClient):
    """Client for OpenAI SDK calls."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError as e:
                raise LlmModelError(f"OpenAI SDK not available: {e}")
        return self._client

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        temperature: float,
        **kwargs,
    ) -> Tuple[str, str]:
        """Make a chat completion call using OpenAI SDK."""
        client = self._get_client()

        try:
            # GPT-5 models (and variants) currently do not support custom
            # temperature values; only the default is allowed. Omit the
            # temperature parameter for these models to avoid 400 errors.
            is_gpt5_series_model = model_name.lower().startswith("gpt-5")

            # Remove unsupported/unknown parameters for chat.completions
            # (e.g., a "reasoning" dict intended for Responses API)
            kwargs.pop("reasoning", None)

            request_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }

            if not is_gpt5_series_model:
                request_kwargs["temperature"] = temperature

            response = client.chat.completions.create(**request_kwargs, **kwargs)

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI")

            if response.choices[0].finish_reason == "length":
                module_logger.warning("Response truncated: consider increasing max_tokens or reviewing model limits")

            return content, response.id

        except Exception as e:
            module_logger.error(f"OpenAI API call failed: {e}")
            raise LlmModelError(f"OpenAI API call failed: {e}")


class GeminiClient(ProviderClient):
    """Client for Google Gemini SDK calls."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError as e:
                raise LlmModelError(f"Google GenAI SDK not available: {e}")
        return self._client

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        temperature: float,
        **kwargs,
    ) -> Tuple[str, str]:
        """Make a chat completion call using Gemini SDK."""
        client = self._get_client()

        try:
            # Configure generation settings
            # Remove unsupported parameters (e.g., reasoning)
            kwargs.pop("reasoning", None)
            generation_config = {"temperature": temperature, **kwargs}

            model = client.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt,
                generation_config=generation_config,
            )

            response = model.generate_content(user_prompt)

            if not response.text:
                raise ValueError("Empty response from Gemini")

            # Gemini doesn't provide a completion ID in the same way, so we'll generate one
            generation_id = f"gemini_{int(time.time())}"

            return response.text, generation_id

        except Exception as e:
            module_logger.error(f"Gemini API call failed: {e}")
            raise LlmModelError(f"Gemini API call failed: {e}")


class LlmModel:
    """
    Unified LLM client that supports multiple providers (OpenRouter, OpenAI, Gemini).
    Routes requests to the appropriate provider based on model configuration.
    """

    DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_OPENROUTER_API_ENV = "OPENROUTER_API_KEY"
    DEFAULT_OPENAI_API_ENV = "OPENAI_API_KEY"
    DEFAULT_GEMINI_API_ENV = "GEMINI_API_KEY"
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0  # seconds
    DEFAULT_BACKOFF_FACTOR = 2.0

    def __init__(
        self,
        model: ModelConfig,
        temperature: float = 0.2,
        # OpenRouter settings
        openrouter_api_key_env: str = DEFAULT_OPENROUTER_API_ENV,
        openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        openrouter_max_retries: int = DEFAULT_MAX_RETRIES,
        openrouter_retry_delay: float = DEFAULT_RETRY_DELAY,
        openrouter_backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        # Provider-specific API key environments
        openai_api_key_env: str = DEFAULT_OPENAI_API_ENV,
        gemini_api_key_env: str = DEFAULT_GEMINI_API_ENV,
    ) -> None:
        """
        Initialize LLM client with provider routing capabilities.

        Args:
            model: Model configuration (ModelConfig) specifying provider and model.
            temperature: Temperature for sampling.
            openrouter_api_key_env: Environment variable for OpenRouter API key.
            openrouter_base_url: OpenRouter base URL.
            openrouter_max_retries: Maximum retry attempts for OpenRouter.
            openrouter_retry_delay: Initial retry delay for OpenRouter.
            openrouter_backoff_factor: Backoff multiplier for OpenRouter.
            openai_api_key_env: Environment variable for OpenAI API key.
            gemini_api_key_env: Environment variable for Gemini API key.
        """
        self.model_config = model
        self.temperature = temperature

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

        if provider == Provider.OPENROUTER:
            api_key_env = self._config["openrouter"]["api_key_env"]
        elif provider == Provider.OPENAI:
            api_key_env = self._config["openai"]["api_key_env"]
        elif provider == Provider.GEMINI:
            api_key_env = self._config["gemini"]["api_key_env"]
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if not os.getenv(api_key_env):
            msg = f"Missing environment variable: {api_key_env}"
            module_logger.error(msg)
            raise ValueError(msg)

    def _get_client(self) -> ProviderClient:
        """Get or create the appropriate provider client."""
        provider = self.model_config.provider

        if provider not in self._clients:
            if provider == Provider.OPENROUTER:
                config = self._config["openrouter"]
                api_key = os.getenv(config["api_key_env"])
                if api_key is None:
                    raise ValueError(f"Missing environment variable: {config['api_key_env']}")
                self._clients[provider] = OpenRouterClient(
                    api_key=api_key,
                    base_url=config["base_url"],
                    max_retries=config["max_retries"],
                    retry_delay=config["retry_delay"],
                    backoff_factor=config["backoff_factor"],
                )
            elif provider == Provider.OPENAI:
                api_key = os.getenv(self._config["openai"]["api_key_env"])
                if api_key is None:
                    raise ValueError(f"Missing environment variable: {self._config['openai']['api_key_env']}")
                self._clients[provider] = OpenAIClient(api_key=api_key)
            elif provider == Provider.GEMINI:
                api_key = os.getenv(self._config["gemini"]["api_key_env"])
                if api_key is None:
                    raise ValueError(f"Missing environment variable: {self._config['gemini']['api_key_env']}")
                self._clients[provider] = GeminiClient(api_key=api_key)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        return self._clients[provider]

    @classmethod
    def create(
        cls,
        model: ModelConfig,
        temperature: float = 0.2,
        openrouter_api_key_env: str = DEFAULT_OPENROUTER_API_ENV,
        openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        openrouter_max_retries: int = DEFAULT_MAX_RETRIES,
        openrouter_retry_delay: float = DEFAULT_RETRY_DELAY,
        openrouter_backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        openai_api_key_env: str = DEFAULT_OPENAI_API_ENV,
        gemini_api_key_env: str = DEFAULT_GEMINI_API_ENV,
    ) -> "LlmModel":
        """Create a new LlmModel instance with the specified configuration."""
        return cls(
            model=model,
            temperature=temperature,
            openrouter_api_key_env=openrouter_api_key_env,
            openrouter_base_url=openrouter_base_url,
            openrouter_max_retries=openrouter_max_retries,
            openrouter_retry_delay=openrouter_retry_delay,
            openrouter_backoff_factor=openrouter_backoff_factor,
            openai_api_key_env=openai_api_key_env,
            gemini_api_key_env=gemini_api_key_env,
        )

    @property
    def model_id(self) -> str:
        """Get the model ID for backward compatibility."""
        return self.model_config.model_id

    def __str__(self) -> str:
        return (
            f"LlmModel(provider={self.model_config.provider.value}, "
            f"model_id={self.model_config.model_id}, temperature={self.temperature})"
        )

    def __repr__(self) -> str:
        return (
            f"LlmModel(provider={self.model_config.provider.value}, "
            f"model_id={self.model_config.model_id}, temperature={self.temperature})"
        )

    def _log_prompt(self, role: str, content: str) -> None:
        """
        Logs a preview of the prompt content.

        Args:
            role (str): The role of the prompt (e.g., "System", "User").
            content (str): The full content of the prompt.
        """
        preview = content if len(content) <= 200 else content[:200] + "..."
        module_logger.trace(f"{role} prompt: {preview}")

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Make a chat completion call using the appropriate provider.

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

        # Allow per-call temperature override without duplicating keyword
        call_temperature: float = kwargs.pop("temperature", self.temperature)

        return client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=model_name,
            temperature=call_temperature,
            **kwargs,
        )
