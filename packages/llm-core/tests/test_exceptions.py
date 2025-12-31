"""Tests for llm_core exceptions module."""

from llm_core.exceptions import (
    APIError,
    AuthenticationError,
    GenerationFailedError,
    LlmModelError,
    MaxRetriesExceededError,
    RateLimitError,
    ResponseTruncatedError,
)


class TestLlmModelError:
    """Tests for base LlmModelError."""

    def test_is_exception(self):
        """Test that LlmModelError is an Exception."""
        error = LlmModelError("test error")
        assert isinstance(error, Exception)

    def test_message(self):
        """Test that message is preserved."""
        error = LlmModelError("test message")
        assert str(error) == "test message"


class TestGenerationFailedError:
    """Tests for GenerationFailedError."""

    def test_inherits_from_llm_model_error(self):
        """Test inheritance from LlmModelError."""
        error = GenerationFailedError("generation failed")
        assert isinstance(error, LlmModelError)

    def test_message_attribute(self):
        """Test message attribute."""
        error = GenerationFailedError("test message")
        assert error.message == "test message"

    def test_block_info_default_none(self):
        """Test block_info defaults to None."""
        error = GenerationFailedError("test message")
        assert error.block_info is None

    def test_block_info_provided(self):
        """Test block_info when provided."""
        error = GenerationFailedError("test message", block_info="block 1")
        assert error.block_info == "block 1"


class TestMaxRetriesExceededError:
    """Tests for MaxRetriesExceededError."""

    def test_inherits_from_llm_model_error(self):
        """Test inheritance from LlmModelError."""
        error = MaxRetriesExceededError("max retries", attempts=3)
        assert isinstance(error, LlmModelError)

    def test_attributes(self):
        """Test all attributes are stored."""
        error = MaxRetriesExceededError("max retries exceeded", attempts=5, block_info="block 2")
        assert error.message == "max retries exceeded"
        assert error.attempts == 5
        assert error.block_info == "block 2"

    def test_block_info_default_none(self):
        """Test block_info defaults to None."""
        error = MaxRetriesExceededError("message", attempts=3)
        assert error.block_info is None


class TestResponseTruncatedError:
    """Tests for ResponseTruncatedError."""

    def test_inherits_from_llm_model_error(self):
        """Test inheritance from LlmModelError."""
        error = ResponseTruncatedError("truncated")
        assert isinstance(error, LlmModelError)

    def test_attributes(self):
        """Test all attributes are stored."""
        error = ResponseTruncatedError("response truncated", model_name="gpt-4")
        assert error.message == "response truncated"
        assert error.model_name == "gpt-4"

    def test_model_name_default_none(self):
        """Test model_name defaults to None."""
        error = ResponseTruncatedError("message")
        assert error.model_name is None


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_inherits_from_llm_model_error(self):
        """Test inheritance from LlmModelError."""
        error = RateLimitError("rate limited")
        assert isinstance(error, LlmModelError)

    def test_message(self):
        """Test message is preserved."""
        error = RateLimitError("too many requests")
        assert str(error) == "too many requests"


class TestAPIError:
    """Tests for APIError."""

    def test_inherits_from_llm_model_error(self):
        """Test inheritance from LlmModelError."""
        error = APIError("api error")
        assert isinstance(error, LlmModelError)

    def test_message(self):
        """Test message is preserved."""
        error = APIError("server error")
        assert str(error) == "server error"


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_inherits_from_llm_model_error(self):
        """Test inheritance from LlmModelError."""
        error = AuthenticationError("invalid key")
        assert isinstance(error, LlmModelError)

    def test_message(self):
        """Test message is preserved."""
        error = AuthenticationError("invalid api key")
        assert str(error) == "invalid api key"
