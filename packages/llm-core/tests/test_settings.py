"""Tests for llm_core settings module."""

import os
from unittest.mock import patch

from llm_core.config import Settings


class TestSettings:
    """Tests for Settings configuration."""

    def test_default_values(self):
        """Test default values are set correctly."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
            assert settings.openai_base_url == "https://api.openai.com/v1"
            assert settings.debug is False
            assert settings.log_level == "INFO"
            assert settings.llm_enable_prompt_logging is False

    def test_api_keys_from_environment(self):
        """Test API keys are loaded from environment."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "or-key",
                "OPENAI_API_KEY": "oai-key",
                "GEMINI_API_KEY": "gem-key",
                "ANTHROPIC_API_KEY": "ant-key",
            },
        ):
            settings = Settings()
            assert settings.openrouter_api_key.get_secret_value() == "or-key"
            assert settings.openai_api_key.get_secret_value() == "oai-key"
            assert settings.gemini_api_key.get_secret_value() == "gem-key"
            assert settings.anthropic_api_key.get_secret_value() == "ant-key"

    def test_api_keys_none_when_not_set(self):
        """Test API keys are None when not set in environment (ignoring .env file)."""
        with patch.dict(os.environ, {}, clear=True):
            # Disable .env file reading to test pure environment variable behavior
            settings = Settings(_env_file=None)
            assert settings.openrouter_api_key is None
            assert settings.openai_api_key is None
            assert settings.gemini_api_key is None
            assert settings.anthropic_api_key is None


class TestGetApiKey:
    """Tests for Settings.get_api_key method."""

    def test_get_openrouter_key(self):
        """Test getting OpenRouter API key."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-or-key"}):
            settings = Settings()
            assert settings.get_api_key("openrouter") == "test-or-key"
            assert settings.get_api_key("OPENROUTER") == "test-or-key"

    def test_get_openai_key(self):
        """Test getting OpenAI API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-oai-key"}):
            settings = Settings()
            assert settings.get_api_key("openai") == "test-oai-key"
            assert settings.get_api_key("OpenAI") == "test-oai-key"

    def test_get_gemini_key(self):
        """Test getting Gemini API key."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gem-key"}):
            settings = Settings()
            assert settings.get_api_key("gemini") == "test-gem-key"

    def test_get_anthropic_key(self):
        """Test getting Anthropic API key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-ant-key"}):
            settings = Settings()
            assert settings.get_api_key("anthropic") == "test-ant-key"

    def test_get_claude_key_returns_anthropic(self):
        """Test that 'claude' returns Anthropic key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-ant-key"}):
            settings = Settings()
            assert settings.get_api_key("claude") == "test-ant-key"

    def test_get_unknown_provider_returns_none(self):
        """Test unknown provider returns None."""
        with patch.dict(os.environ, {}):
            settings = Settings()
            assert settings.get_api_key("unknown") is None

    def test_get_api_key_case_insensitive(self):
        """Test get_api_key is case insensitive."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            assert settings.get_api_key("openai") == "test-key"
            assert settings.get_api_key("OpenAI") == "test-key"
            assert settings.get_api_key("OPENAI") == "test-key"


class TestGetEnv:
    """Tests for Settings.get_env method."""

    def test_get_existing_env(self):
        """Test getting existing environment variable."""
        with patch.dict(os.environ, {"MY_VAR": "my_value"}):
            settings = Settings()
            assert settings.get_env("MY_VAR") == "my_value"

    def test_get_nonexistent_env_default_none(self):
        """Test getting nonexistent env returns None by default."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.get_env("NONEXISTENT_VAR") is None

    def test_get_nonexistent_env_with_default(self):
        """Test getting nonexistent env with custom default."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.get_env("NONEXISTENT_VAR", "default_value") == "default_value"
