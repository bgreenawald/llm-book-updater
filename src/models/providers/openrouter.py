import json
import time
from typing import Any, Tuple

import requests  # type: ignore[import-untyped]
from loguru import logger
from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]
from urllib3.util.retry import Retry

from src.api.api_models import OpenRouterResponse
from src.core.constants import (
    DEFAULT_OPENROUTER_BACKOFF_FACTOR,
    DEFAULT_OPENROUTER_MAX_RETRIES,
    DEFAULT_OPENROUTER_RETRY_DELAY,
    OPENROUTER_POOL_CONNECTIONS,
    OPENROUTER_POOL_MAXSIZE,
    OPENROUTER_REQUEST_TIMEOUT,
)
from src.models.base import ProviderClient
from src.models.exceptions import LlmModelError, ResponseTruncatedError

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
