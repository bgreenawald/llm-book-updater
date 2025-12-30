"""OpenRouter API client with retry logic."""

from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .models import GenerationConfig


class OpenRouterError(Exception):
    """Base exception for OpenRouter errors."""

    pass


class RateLimitError(OpenRouterError):
    """Rate limit exceeded."""

    pass


class APIError(OpenRouterError):
    """General API error."""

    pass


class AuthenticationError(OpenRouterError):
    """Authentication failed."""

    pass


class OpenRouterClient:
    """Async client for OpenRouter API with retry logic."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, config: GenerationConfig):
        self.api_key = api_key
        self.config = config
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(
        self,
        messages: list[dict],
        model: Optional[str] = None,
    ) -> str:
        """
        Generate completion with automatic retry logic.
        Uses tenacity for exponential backoff.
        """
        model = model or self.config.model

        try:
            response = await self._call_api_with_retry(messages, model)
            return self._extract_content(response)
        except Exception as e:
            # Re-raise as OpenRouterError if not already
            if isinstance(e, OpenRouterError):
                raise
            raise APIError(f"Unexpected error: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((RateLimitError, httpx.TimeoutException)),
        reraise=True,
    )
    async def _call_api_with_retry(
        self,
        messages: list[dict],
        model: str,
    ) -> dict:
        """Make single API call with retry wrapper."""
        return await self._call_api(messages, model)

    async def _call_api(
        self,
        messages: list[dict],
        model: str,
    ) -> dict:
        """Make a single API call to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/non-fiction-book-writer",
            "X-Title": "Non-Fiction Book Writer",
        }

        payload = {"model": model, "messages": messages, "reasoning": {"effort": "high"}}

        try:
            response = await self.client.post(
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
            # Server errors - retry
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

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


async def create_client(api_key: str, config: GenerationConfig) -> OpenRouterClient:
    """Create and return an OpenRouter client."""
    return OpenRouterClient(api_key, config)
