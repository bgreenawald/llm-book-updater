import os
from abc import ABC

from openai import OpenAI


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
        api_key = os.getenv(api_key_environment_variable_name)
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_environment_variable_name} is not set"
            )
        self.base_url = base_url
        self.model_id = model_id

        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    def chat_completion(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
        if not response or not response.choices or not response.choices[0].message:
            raise ValueError("Invalid response from the API")

        return response.choices[0].message.content


class Grok3Mini(LlmModel):
    def __init__(self):
        super().__init__(
            api_key_environment_variable_name="GROK_API_KEY",
            model_id="grok-3-mini",
            base_url="https://api.x.ai/v1",
        )
