import json
import os

import requests
from loguru import logger

from src.logging_config import setup_logging

# Initialize module‐level logger
logger = setup_logging("llm_model")

# Model constants
GROK_3_MINI = "x-ai/grok-3-mini"
GEMINI_FLASH = "google/gemini-2.5-flash"
GEMINI_PRO = "google/gemini-2.5-pro"
DEEPSEEK = "deepseek/deepseek-r1-0528"
OPENAI_04_MINI = "openai/o4-mini-high"
CLAUDE_4_SONNET = "anthropic/claude-sonnet-4"
GEMINI_FLASH_LITE = "google/gemini-2.5-flash-lite-preview-06-17"


class LlmModel:
    """
    LLM client backed by OpenRouter. Makes direct API calls to OpenRouter
    with logging and validation.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_API_ENV = "OPENROUTER_API_KEY"

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        api_key_env: str = DEFAULT_API_ENV,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        """
        Args:
            model:        Which model to call (string model identifier).
            temperature:  Temperature for sampling.
            api_key_env:  Name of the ENV var holding your OpenRouter API key.
            base_url:     OpenRouter base endpoint.
        """
        logger.info(
            f"Initializing LLM client: model={model}, base_url={base_url}"
        )
        api_key = os.getenv(api_key_env)
        if not api_key:
            msg = f"Missing environment variable: {api_key_env}"
            logger.error(msg)
            raise ValueError(msg)

        self.model_id = model
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key
        logger.success(f"LLM client ready: {self.model_id}")

    @classmethod
    def create(
        cls,
        model: str,
        temperature: float = 0.2,
        api_key_env: str = DEFAULT_API_ENV,
        base_url: str = DEFAULT_BASE_URL,
    ) -> "LlmModel":
        """Create a new LlmModel instance with the specified configuration."""
        return cls(
            model=model,
            temperature=temperature,
            api_key_env=api_key_env,
            base_url=base_url,
        )

    def __str__(self):
        return f"LlmModel(model_id={self.model_id}, temperature={self.temperature})"

    def __repr__(self):
        return f"LlmModel(model_id={self.model_id}, temperature={self.temperature})"

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
        Make a chat completion call using OpenRouter API.

        Returns:
            assistant reply content.

        Raises:
            ValueError on empty/malformed response.
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

        # Make the API call
        try:
            response = requests.post(
                url=f"{self.base_url}/chat/completions",
                headers=headers,
                data=json.dumps(data),
            )
            response.raise_for_status()

            resp_data = response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise ValueError(f"API request failed: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response JSON: {e}")
            raise ValueError(f"Failed to parse response JSON: {e}")

        # Extract response content
        choices = resp_data.get("choices", [])
        if not choices or not choices[0].get("message", {}).get("content"):
            raise ValueError(f"Empty or malformed response: {resp_data}")

        content = choices[0]["message"]["content"]
        finish_reason = choices[0].get("finish_reason", "unknown")
        usage = resp_data.get("usage", {})

        details = {
            "id": resp_data.get("id", ""),
            "model": resp_data.get("model", self.model_id),
            "finish_reason": finish_reason,
        }

        if finish_reason == "length":
            logger.warning(
                "Response truncated: consider increasing max_tokens or reviewing model limits"
            )

        return content
