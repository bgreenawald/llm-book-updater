import os
from abc import ABC

from loguru import logger
from openai import OpenAI

from src.logging_config import setup_logging

# Set up logging
logger = setup_logging("llm_model")


class LlmModel(ABC):
    def __init__(
        self,
        api_key_environment_variable_name: str,
        model_id: str,
        base_url: str,
    ) -> None:
        """
        Initializes the LLM model with the given base URL, model ID, and API key environment variable name.

        Args:
            api_key_environment_variable_name (str): The name of the environment variable containing the API key.
            model_id (str): The ID of the model.
            base_url (str): The base URL of the API.

        Raises:
            ValueError: If the API key environment variable is not set.
        """
        logger.info(
            f"Initializing LLM model with base URL: {base_url}, model ID: {model_id}"
        )

        api_key = os.getenv(api_key_environment_variable_name)
        if not api_key:
            error_msg = (
                f"Environment variable {api_key_environment_variable_name} is not set"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.base_url = base_url
        self.model_id = model_id
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        logger.success(f"Successfully initialized LLM model: {self.model_id}")

    def chat_completion(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        logger.debug(f"Starting chat completion with model: {self.model_id}")
        logger.trace(
            f"System prompt: {system_prompt[:200]}..."
            if len(system_prompt) > 200
            else f"System prompt: {system_prompt}"
        )
        logger.trace(
            f"User prompt: {user_prompt[:200]}..."
            if len(user_prompt) > 200
            else f"User prompt: {user_prompt}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **kwargs,
            )

            if not response or not response.choices or not response.choices[0].message:
                error_msg = "Received invalid response from the API"
                logger.error(error_msg)
                raise ValueError(error_msg)

            result = response.choices[0].message.content
            logger.debug(
                f"Successfully completed chat completion. Response length: {len(result)} characters"
            )
            return result

        except Exception as e:
            logger.error(f"Error during chat completion: {str(e)}")
            raise


class Grok3Mini(LlmModel):
    def __init__(self):
        super().__init__(
            api_key_environment_variable_name="GROK_API_KEY",
            model_id="grok-3-mini",
            base_url="https://api.x.ai/v1",
        )
        logger.info("Initialized Grok3Mini model")


class Gemini2Flash(LlmModel):
    def __init__(self):
        super().__init__(
            api_key_environment_variable_name="GEMINI_API_KEY",
            model_id="gemini-2.5-flash-preview-05-20",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        logger.info("Initialized Gemini2Flash model")


class Gemini2Pro(LlmModel):
    def __init__(self):
        super().__init__(
            api_key_environment_variable_name="GEMINI_API_KEY",
            model_id="gemini-2.5-pro-preview-05-06",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        logger.info("Initialized Gemini2Pro model")
