"""Pytest configuration for book-writer tests."""

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
def sample_rubric_content():
    """Sample rubric content for testing."""
    return """# My Test Book

## Introduction

Some intro content that should be included in the first section.

### Opening Vignette

A brief opening story.

### Core Concepts

#### Key Ideas

Explain the main ideas.

#### Examples

Provide examples.

## Advanced Topics

### Deep Dive

Going deeper.

### Applications

Real-world applications.
"""


@pytest.fixture
def sample_rubric_file(temp_dir, sample_rubric_content):
    """Create a sample rubric file."""
    rubric_file = temp_dir / "rubric.md"
    rubric_file.write_text(sample_rubric_content)
    return rubric_file


@pytest.fixture
def output_dir(temp_dir):
    """Create an output directory."""
    out_dir = temp_dir / "output"
    out_dir.mkdir()
    return out_dir
