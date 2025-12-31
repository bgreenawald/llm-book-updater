"""Custom exceptions for LLM Core library."""

from typing import Optional


class LlmModelError(Exception):
    """Custom exception for LLM model errors."""

    pass


class GenerationFailedError(LlmModelError):
    """Exception raised when an LLM generation fails and cannot be retried.

    This is raised when:
    - Retry is disabled and a generation fails
    - A generation produces an empty or error response

    Attributes:
        message: Description of the failure
        block_info: Optional information about the block that failed
    """

    def __init__(self, message: str, block_info: Optional[str] = None):
        self.message = message
        self.block_info = block_info
        super().__init__(self.message)


class MaxRetriesExceededError(LlmModelError):
    """Exception raised when maximum retries are exhausted for a generation.

    This signals that the pipeline should stop because a generation could not
    succeed after the configured number of retry attempts.

    Attributes:
        message: Description of the failure
        attempts: Number of attempts made
        block_info: Optional information about the block that failed
    """

    def __init__(self, message: str, attempts: int, block_info: Optional[str] = None):
        self.message = message
        self.attempts = attempts
        self.block_info = block_info
        super().__init__(self.message)


class ResponseTruncatedError(LlmModelError):
    """Exception raised when an LLM response is truncated due to max_tokens limit.

    This is a retryable error - the generation may succeed on retry if the model
    uses less reasoning tokens or produces a shorter response.

    Attributes:
        message: Description of the truncation
        model_name: The model that produced the truncated response
    """

    def __init__(self, message: str, model_name: Optional[str] = None):
        self.message = message
        self.model_name = model_name
        super().__init__(self.message)


# Async-specific exceptions (used by async providers)


class RateLimitError(LlmModelError):
    """Rate limit exceeded - retryable with backoff."""

    pass


class APIError(LlmModelError):
    """General API error - may or may not be retryable."""

    pass


class AuthenticationError(LlmModelError):
    """Authentication failed - not retryable."""

    pass
