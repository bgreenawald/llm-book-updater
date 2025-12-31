"""Pytest configuration for book-updater tests."""

import os
import tempfile
from pathlib import Path
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


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_input_file(temp_dir):
    """Create a sample input file for testing."""
    input_file = temp_dir / "input.md"
    input_file.write_text("# Chapter 1\n\nSome content here.\n\n## Section 1.1\n\nMore content.")
    return input_file


@pytest.fixture
def sample_output_dir(temp_dir):
    """Create an output directory for testing."""
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    return output_dir
