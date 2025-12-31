"""OpenRouter provider clients (sync and async) with connection pooling and retry logic."""

import json
import time
from typing import Any, Optional, Tuple

import httpx
import requests  # type: ignore[import-untyped]
from loguru import logger
from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from urllib3.util.retry import Retry

from llm_core.api_models import OpenRouterResponse
from llm_core.config import (
    DEFAULT_OPENROUTER_BACKOFF_FACTOR,
    DEFAULT_OPENROUTER_MAX_RETRIES,
    DEFAULT_OPENROUTER_RETRY_DELAY,
    OPENROUTER_POOL_CONNECTIONS,
    OPENROUTER_POOL_MAXSIZE,
    OPENROUTER_REQUEST_TIMEOUT,
)
from llm_core.exceptions import (
    APIError,
    AuthenticationError,
    LlmModelError,
    RateLimitError,
    ResponseTruncatedError,
)
from llm_core.providers.base import AsyncProviderClient, ProviderClient

module_logger = logger


class OpenRouterClient(ProviderClient):
    """Client for OpenRouter API calls with connection pooling."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = DEFAULT_OPENROUTER_MAX_RETRIES,
        retry_delay: float = DEFAULT_OPENROUTER_RETRY_DELAY,
        backoff_factor: float = DEFAULT_OPENROUTER_BACKOFF_FACTOR,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

        # Initialize session with connection pooling
        self._session = requests.Session()

        # Configure retry strategy for the session
        # Note: This handles automatic retries at the connection level
        # We still keep our application-level retry logic for custom backoff
        retry_strategy = Retry(
            total=0,  # Disable automatic retries; we handle retries manually
            connect=3,  # But allow connection retries for network issues
            read=3,  # And read retries for incomplete responses
            status_forcelist=[500, 502, 503, 504],  # Retry on server errors
        )

        # Configure connection pooling
        adapter = HTTPAdapter(
            pool_connections=OPENROUTER_POOL_CONNECTIONS,
            pool_maxsize=OPENROUTER_POOL_MAXSIZE,
            max_retries=retry_strategy,
        )

        # Mount adapter for both http and https
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        # Set default headers for all requests
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def close(self) -> None:
        """Close the session and release connection pool resources."""
        if hasattr(self, "_session"):
            self._session.close()

    @property
    def supports_batch(self) -> bool:
        """OpenRouter does not support batch processing."""
        return False

    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry."""
        if isinstance(error, requests.exceptions.RequestException):
            if hasattr(error, "response") and error.response is not None:
                status_code = error.response.status_code
                return status_code >= 500 or status_code == 429
            return True
        return False

    def _make_api_call(self, data: dict) -> dict:
        """Makes a single API call to OpenRouter with retry logic using session.

        Args:
            data: Request payload to send to the API

        Returns:
            Response JSON from the API

        Note:
            Headers are already set in the session, so we don't need to pass them.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.post(
                    url=f"{self.base_url}/chat/completions",
                    data=json.dumps(obj=data),
                    timeout=OPENROUTER_REQUEST_TIMEOUT,
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
        raise LlmModelError(error_msg) from last_error

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        **kwargs,
    ) -> Tuple[str, str]:
        """Make a chat completion call using OpenRouter API."""
        data: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        }

        resp_data = self._make_api_call(data=data)
        response = OpenRouterResponse.model_validate(resp_data)

        if not response.choices or not response.choices[0].message.get("content"):
            # Extract additional diagnostic information from raw response
            choice_data = resp_data.get("choices", [{}])[0] if resp_data.get("choices") else {}
            finish_reason = choice_data.get("finish_reason", "unknown")
            native_finish_reason = choice_data.get("native_finish_reason")
            usage = resp_data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)

            # Build a more informative error message
            error_parts = ["Empty or malformed response"]
            if native_finish_reason:
                error_parts.append(f"native_finish_reason: {native_finish_reason}")
            if finish_reason and finish_reason != "unknown":
                error_parts.append(f"finish_reason: {finish_reason}")
            if completion_tokens == 0:
                error_parts.append("no tokens generated")

            error_msg = ", ".join(error_parts)
            if native_finish_reason == "abort":
                error_msg += " (generation was aborted by provider/model)"

            raise ValueError(f"{error_msg}: {resp_data}")

        content = response.choices[0].message["content"]
        finish_reason = response.choices[0].finish_reason or "unknown"

        if finish_reason == "length":
            module_logger.warning("Response truncated: consider increasing max_tokens or reviewing model limits")
            raise ResponseTruncatedError(
                message="Response truncated due to max_tokens limit",
                model_name=model_name,
            )

        generation_id = response.id or "unknown"
        return content, generation_id


class AsyncOpenRouterClient(AsyncProviderClient):
    """Async client for OpenRouter API with retry logic.

    Uses httpx for async HTTP requests and tenacity for exponential backoff.
    Suitable for parallel/concurrent LLM generation workflows.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        default_model: str = "anthropic/claude-sonnet-4",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """Initialize the async OpenRouter client.

        Args:
            api_key: OpenRouter API key
            default_model: Default model to use if not specified per-request
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for rate limits/timeouts
        """
        self.api_key = api_key
        self.default_model = default_model
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout)

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
    ) -> str:
        """Generate completion with automatic retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Optional model override (uses default_model if not specified)

        Returns:
            Generated content string

        Raises:
            APIError: On non-retryable API errors
            AuthenticationError: On authentication failure
            RateLimitError: If rate limits persist after retries
        """
        model = model or self.default_model

        try:
            response = await self._call_api_with_retry(messages, model)
            return self._extract_content(response)
        except (RateLimitError, AuthenticationError, APIError):
            raise
        except Exception as e:
            raise APIError(f"Unexpected error: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((RateLimitError, httpx.TimeoutException)),
        reraise=True,
    )
    async def _call_api_with_retry(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> dict:
        """Make single API call with retry wrapper."""
        return await self._call_api(messages, model)

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> dict:
        """Make a single API call to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/llm-book-updater",
            "X-Title": "LLM Book Tools",
        }

        payload = {
            "model": model,
            "messages": messages,
            "reasoning": {"effort": "high"},
        }

        try:
            response = await self._client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
        except httpx.TimeoutException:
            raise  # Let tenacity retry this

        # Handle response status codes
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        elif response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        elif response.status_code >= 500:
            # Server errors - retry via RateLimitError
            raise RateLimitError(f"Server error: {response.status_code}")
        else:
            # Client errors - don't retry
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", str(error_data))
            except Exception:
                error_msg = response.text
            raise APIError(f"API error ({response.status_code}): {error_msg}")

    def _extract_content(self, response: dict) -> str:
        """Extract generated content from API response."""
        try:
            choices = response.get("choices", [])
            if not choices:
                raise APIError("No choices in response")

            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content")

            # Content can be null/empty for tool_calls, refusals, or reasoning models
            if content:
                return content

            # Check for refusal
            if message.get("refusal"):
                raise APIError(f"Model refused: {message['refusal']}")

            # Check for tool calls (content is expected to be null)
            if message.get("tool_calls"):
                raise APIError("Response contains tool_calls but no content")

            # Check finish_reason for more context
            finish_reason = choice.get("finish_reason")
            if finish_reason == "content_filter":
                raise APIError("Content filtered by provider")
            elif finish_reason == "error":
                raise APIError("Model returned an error")
            elif finish_reason == "length":
                raise APIError("Response truncated due to length limit")

            raise APIError(f"Empty content in response (finish_reason: {finish_reason})")
        except KeyError as e:
            raise APIError(f"Unexpected response format: {e}")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
