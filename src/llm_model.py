import os
import time
from abc import ABC

from loguru import logger
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from src.logging_config import setup_logging

# Initialize module-level logger
logger = setup_logging("llm_model")


class LlmModel(ABC):
    """
    Abstract base class for LLM clients, wraps OpenAI chat completion calls
    with retries, logging, and response validation.
    """

    def __init__(
        self,
        api_key_env: str,
        model_id: str,
        base_url: str,
    ) -> None:
        """
        Args:
            api_key_env: Name of the environment variable holding the API key.
            model_id: Identifier of the model to use.
            base_url: Base API URL.

        Raises:
            ValueError: If the API key is missing.
        """
        logger.info(f"Initializing LLM client: model={model_id}, base_url={base_url}")
        api_key = os.getenv(api_key_env)
        if not api_key:
            msg = f"Missing environment variable: {api_key_env}"
            logger.error(msg)
            raise ValueError(msg)

        self.model_id = model_id
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.success(f"LLM client ready: {model_id}")

    def _log_prompt(self, role: str, content: str) -> None:
        """Log a truncated version of the prompt for debugging."""
        preview = content if len(content) <= 200 else content[:200] + "..."
        logger.trace(f"{role} prompt: {preview}")

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        """
        Call the chat completion endpoint with automatic retries on transient errors.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User message content.
            max_retries: Number of retry attempts for transient failures.
            **kwargs: Additional parameters for the API (e.g., max_tokens, temperature).

        Returns:
            The assistant's reply content as a string.

        Raises:
            ValueError: On invalid or empty API responses.
            APIError: On non-recoverable API failures.
        """
        logger.debug(f"Starting chat_completions for model={self.model_id}")
        self._log_prompt("System", system_prompt)
        self._log_prompt("User", user_prompt)

        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **kwargs,
                )

                # Basic validation of response structure
                choices = getattr(resp, "choices", None)
                if not choices or not choices[0].message.content:
                    raise ValueError(f"Empty or malformed response: {resp}")

                content = choices[0].message.content
                finish_reason = choices[0].finish_reason or "unknown"
                usage = getattr(resp, "usage", {})

                # Log usage and content preview
                details = {
                    "id": getattr(resp, "id", ""),
                    "model": getattr(resp, "model", self.model_id),
                    "finish_reason": finish_reason,
                    "tokens": {
                        "prompt": usage.get("prompt_tokens", 0),
                        "completion": usage.get("completion_tokens", 0),
                        "total": usage.get("total_tokens", 0),
                    },
                }
                logger.debug(
                    "Chat completion succeeded",
                    extra={"details": details, "preview": content[:200]},
                )

                if finish_reason == "length":
                    logger.warning(
                        "Response truncated: consider increasing max_tokens or reviewing model limits"
                    )

                return content

            except (APITimeoutError, APIConnectionError) as e:
                last_exc = e
                if attempt < max_retries:
                    delay = 2**attempt + 1
                    logger.warning(
                        f"Connection error (try {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s"
                    )
                    time.sleep(delay)
                    continue
                logger.error(f"Connection failed after retries: {e}")
                raise

            except RateLimitError as e:
                last_exc = e
                retry_after = int(getattr(e.response.headers, "Retry-After", 5))
                if attempt < max_retries:
                    logger.warning(
                        f"Rate limit (try {attempt + 1}/{max_retries}): retrying after {retry_after}s"
                    )
                    time.sleep(retry_after)
                    continue
                logger.error(f"Rate limited after retries: {e}")
                raise

            except APIError as e:
                logger.error(f"API error: {e}", exc_info=True)
                raise

            except Exception as e:
                last_exc = e
                logger.error(
                    f"Unexpected error on attempt {attempt + 1}: {e}", exc_info=True
                )
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                raise

        raise RuntimeError(
            f"Failed after {max_retries} attempts, last error: {last_exc}"
        )


# Concrete model clients
class Grok3Mini(LlmModel):
    def __init__(self):
        super().__init__("GROK_API_KEY", "grok-3-mini", "https://api.x.ai/v1")
        logger.info("Grok3Mini ready")


class Gemini2Flash(LlmModel):
    def __init__(self):
        super().__init__(
            "GEMINI_API_KEY",
            "gemini-2.5-flash-preview-05-20",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        logger.info("Gemini2Flash ready")


class Gemini2Pro(LlmModel):
    def __init__(self):
        super().__init__(
            "GEMINI_API_KEY",
            "gemini-2.5-pro-preview-05-06",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        logger.info("Gemini2Pro ready")
