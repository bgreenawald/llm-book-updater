import json
import os
import time

import requests

from src.logging_config import setup_logging

# Initialize module-level logger
module_logger = setup_logging("llm_model")

# Model constants
GROK_3_MINI = "x-ai/grok-3-mini"
GEMINI_FLASH = "google/gemini-2.5-flash"
GEMINI_PRO = "google/gemini-2.5-pro"
DEEPSEEK = "deepseek/deepseek-r1-0528"
OPENAI_04_MINI = "openai/o4-mini-high"
CLAUDE_4_SONNET = "anthropic/claude-sonnet-4"
GEMINI_FLASH_LITE = "google/gemini-2.5-flash-lite-preview-06-17"


class LlmModelError(Exception):
    """Custom exception for LLM model errors."""

    pass


class LlmModel:
    """
    LLM client backed by OpenRouter. Makes direct API calls to OpenRouter
    with logging and validation.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_API_ENV = "OPENROUTER_API_KEY"
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0  # seconds
    DEFAULT_BACKOFF_FACTOR = 2.0

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        api_key_env: str = DEFAULT_API_ENV,
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ) -> None:
        """
        Args:
            model:         Which model to call (string model identifier).
            temperature:   Temperature for sampling.
            api_key_env:   Name of the ENV var holding your OpenRouter API key.
            base_url:      OpenRouter base endpoint.
            max_retries:   Maximum number of retry attempts for failed API calls.
            retry_delay:   Initial delay between retries in seconds.
            backoff_factor: Multiplier for exponential backoff.
        """
        module_logger.info(f"Initializing LLM client: model={model}, base_url={base_url}")
        api_key = os.getenv(api_key_env)
        if not api_key:
            msg = f"Missing environment variable: {api_key_env}"
            module_logger.error(msg)
            raise ValueError(msg)

        self.model_id = model
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        module_logger.success(f"LLM client ready: {self.model_id}")

    @classmethod
    def create(
        cls,
        model: str,
        temperature: float = 0.2,
        api_key_env: str = DEFAULT_API_ENV,
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ) -> "LlmModel":
        """Create a new LlmModel instance with the specified configuration."""
        return cls(
            model=model,
            temperature=temperature,
            api_key_env=api_key_env,
            base_url=base_url,
            max_retries=max_retries,
            retry_delay=retry_delay,
            backoff_factor=backoff_factor,
        )

    def __str__(self) -> str:
        return f"LlmModel(model_id={self.model_id}, temperature={self.temperature})"

    def __repr__(self) -> str:
        return f"LlmModel(model_id={self.model_id}, temperature={self.temperature})"

    def _log_prompt(self, role: str, content: str) -> None:
        """
        Logs a preview of the prompt content.

        Args:
            role (str): The role of the prompt (e.g., "System", "User").
            content (str): The full content of the prompt.
        """
        preview = content if len(content) <= 200 else content[:200] + "..."
        module_logger.trace(f"{role} prompt: {preview}")

    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry."""
        if isinstance(error, requests.exceptions.RequestException):
            # Retry on network errors, timeouts, and 5xx server errors
            if hasattr(error, "response") and error.response is not None:
                status_code = error.response.status_code
                return status_code >= 500 or status_code == 429
            # Retry on connection errors, timeouts, etc.
            return True
        # Don't retry on JSON decode errors or other non-network issues
        return False

    def _make_api_call(self, headers: dict, data: dict) -> dict:
        """
        Makes a single API call to OpenRouter with retry logic.

        Args:
            headers (dict): HTTP headers for the request.
            data (dict): JSON payload for the request.

        Returns:
            dict: The JSON response from the API.

        Raises:
            LlmModelError: If the API call fails after all retries.
            json.JSONDecodeError: If the response cannot be decoded as JSON.
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url=f"{self.base_url}/chat/completions",
                    headers=headers,
                    data=json.dumps(data),
                    timeout=30,  # Add timeout to prevent hanging requests
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries and self._should_retry(e):
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

        # If we get here, all retries have been exhausted
        error_msg = f"API call failed after {self.max_retries + 1} attempts"
        if last_error:
            error_msg += f": {last_error}"

        module_logger.error(error_msg)
        raise LlmModelError(error_msg)

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:
        """
        Make a chat completion call using OpenRouter API with retry logic.

        Returns:
            assistant reply content.

        Raises:
            LlmModelError: When API calls fail after max retries.
            ValueError: On empty/malformed response.
        """
        self._log_prompt("System", system_prompt)
        self._log_prompt("User", user_prompt)

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Prepare request data
        data = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        }

        # Make the API call with retry logic
        try:
            resp_data = self._make_api_call(headers, data)
        except LlmModelError:
            # Re-raise LlmModelError to stop the pipeline
            raise

        # Extract response content
        choices = resp_data.get("choices", [])
        if not choices or not choices[0].get("message", {}).get("content"):
            raise ValueError(f"Empty or malformed response: {resp_data}")

        content = choices[0]["message"]["content"]
        finish_reason = choices[0].get("finish_reason", "unknown")

        if finish_reason == "length":
            module_logger.warning("Response truncated: consider increasing max_tokens or reviewing model limits")

        return content
