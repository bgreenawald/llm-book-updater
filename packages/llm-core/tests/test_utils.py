"""Tests for llm_core utils module."""

from llm_core.utils import is_failed_response


class TestIsFailedResponse:
    """Tests for is_failed_response utility function."""

    def test_empty_string_is_failed(self):
        """Test that empty string is considered failed."""
        assert is_failed_response("") is True

    def test_none_is_failed(self):
        """Test that None is considered failed."""
        assert is_failed_response(None) is True

    def test_whitespace_only_is_failed(self):
        """Test that whitespace-only string is considered failed."""
        assert is_failed_response("   ") is True
        assert is_failed_response("\t\n") is True
        assert is_failed_response("  \n  \t  ") is True

    def test_error_prefix_is_failed(self):
        """Test that 'Error:' prefix indicates failure."""
        assert is_failed_response("Error: something went wrong") is True
        assert is_failed_response("Error:no space") is True

    def test_error_prefix_with_whitespace_is_failed(self):
        """Test that 'Error:' prefix with leading whitespace is failed."""
        assert is_failed_response("  Error: something went wrong") is True
        assert is_failed_response("\nError: test") is True

    def test_valid_content_not_failed(self):
        """Test that valid content is not considered failed."""
        assert is_failed_response("Hello, world!") is False
        assert is_failed_response("Some valid response") is False

    def test_content_containing_error_word_not_failed(self):
        """Test that content containing 'Error' but not as prefix is not failed."""
        assert is_failed_response("This is not an Error message") is False
        assert is_failed_response("error: lowercase") is False  # Case sensitive

    def test_newlines_and_content_not_failed(self):
        """Test that content with newlines is not failed."""
        assert is_failed_response("Line 1\nLine 2\nLine 3") is False

    def test_single_character_not_failed(self):
        """Test that single character response is not failed."""
        assert is_failed_response("a") is False
        assert is_failed_response("1") is False
