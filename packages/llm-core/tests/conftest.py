"""Pytest configuration for llm-core tests."""

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_api_keys():
    """Mock all API keys for tests."""
    with patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-openrouter-key",
            "OPENAI_API_KEY": "test-openai-key",
            "GEMINI_API_KEY": "test-gemini-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
        },
    ):
        yield
