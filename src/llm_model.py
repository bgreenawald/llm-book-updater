import os
from enum import Enum

from loguru import logger
from openai import OpenAI

from src.logging_config import setup_logging

# Initialize module‐level logger
logger = setup_logging("llm_model")


class ModelType(Enum):
    GROK_3_MINI = "grok-3-mini"
    GEMINI_FLASH = "gemini-2.5-flash-preview-05-20"
    GEMINI_PRO = "gemini-2.5-pro-preview-05-06"


class LlmModel:
    """
    LLM client backed by OpenRouter. Wraps OpenAI-compatible chat completions
    with logging and validation.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/v1"
    DEFAULT_API_ENV = "OPENROUTER_API_KEY"

    def __init__(
        self,
        model: ModelType,
        api_key_env: str = DEFAULT_API_ENV,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        """
        Args:
            model:        Which model to call (from ModelType).
            api_key_env:  Name of the ENV var holding your OpenRouter API key.
            base_url:     OpenRouter base endpoint.
        """
        logger.info(
            f"Initializing LLM client: model={model.value}, base_url={base_url}"
        )
        api_key = os.getenv(api_key_env)
        if not api_key:
            msg = f"Missing environment variable: {api_key_env}"
            logger.error(msg)
            raise ValueError(msg)

        self.model_id = model.value
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.success(f"LLM client ready: {self.model_id}")

    def _log_prompt(self, role: str, content: str) -> None:
        preview = content if len(content) <= 200 else content[:200] + "..."
        logger.trace(f"{role} prompt: {preview}")

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:
        """
        Make a chat completion call.

        Returns:
            assistant reply content.

        Raises:
            ValueError on empty/malformed response.
        """
        logger.debug(f"Starting chat_completions for model={self.model_id}")
        self._log_prompt("System", system_prompt)
        self._log_prompt("User", user_prompt)

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )

        choices = getattr(resp, "choices", None)
        if not choices or not choices[0].message.content:
            raise ValueError(f"Empty or malformed response: {resp}")

        content = choices[0].message.content
        finish_reason = choices[0].finish_reason or "unknown"
        usage = getattr(resp, "usage", {})

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

        return content
