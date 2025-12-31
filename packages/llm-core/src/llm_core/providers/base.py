"""Abstract base classes for provider-specific LLM clients (sync and async)."""

from abc import ABC, abstractmethod
from typing import Any, Tuple


class ProviderClient(ABC):
    """Abstract base class for synchronous provider-specific LLM clients."""

    @property
    @abstractmethod
    def supports_batch(self) -> bool:
        """
        Check if this provider supports batch processing.

        Returns:
            bool: True if batch processing is supported, False otherwise
        """
        pass

    @abstractmethod
    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Make a chat completion call.

        Args:
            system_prompt: System prompt for the conversation
            user_prompt: User prompt for the conversation
            model_name: Name of the model to use
            **kwargs: Additional arguments

        Returns:
            Tuple of (response_content, generation_id)
        """
        pass

    def batch_chat_completion(
        self,
        requests: list[dict[str, Any]],
        model_name: str,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Make batch chat completion calls.

        Args:
            requests: List of request dictionaries, each containing:
                - system_prompt: System prompt for the request
                - user_prompt: User prompt for the request
                - metadata: Optional metadata to include in response
            model_name: Name of the model to use
            **kwargs: Additional arguments

        Returns:
            List of response dictionaries, each containing:
                - content: Response content
                - generation_id: Generation ID for tracking
                - metadata: Original metadata from request
        """
        # Default implementation: process sequentially
        responses = []
        for request in requests:
            content, generation_id = self.chat_completion(
                system_prompt=request["system_prompt"],
                user_prompt=request["user_prompt"],
                model_name=model_name,
                **kwargs,
            )
            responses.append(
                {"content": content, "generation_id": generation_id, "metadata": request.get("metadata", {})}
            )
        return responses


class AsyncProviderClient(ABC):
    """Abstract base class for asynchronous provider-specific LLM clients.

    Async clients support the async context manager protocol for proper
    resource cleanup.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> str:
        """
        Generate a completion from a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Optional model override (uses client default if not specified)

        Returns:
            Generated content string
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the client and release resources."""
        pass

    async def __aenter__(self) -> "AsyncProviderClient":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.close()
