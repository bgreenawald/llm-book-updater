#!/usr/bin/env python3
"""
Pytest configuration and fixtures for the test suite.
"""

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_openrouter_api_key():
    """Mock the OPENROUTER_API_KEY environment variable for all tests."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key-12345"}):
        yield


@pytest.fixture
def mock_llm_model():
    """Mock LlmModel to avoid actual API calls during testing."""
    from unittest.mock import Mock, patch

    from src.llm.model import LlmModel

    with (
        patch.object(LlmModel, "__init__", return_value=None),
        patch.object(LlmModel, "chat_completion", return_value=("Mocked response", "test-gen-123")) as mock_completion,
    ):
        # Create a mock instance with necessary attributes
        mock_instance = Mock(spec=LlmModel)
        mock_instance.model_id = "test/model"
        mock_instance.chat_completion = mock_completion

        with patch("src.llm_model.LlmModel", return_value=mock_instance):
            yield mock_instance


@pytest.fixture
def mock_cost_tracker():
    """Mock cost tracking to avoid API dependencies."""
    from unittest.mock import Mock, patch

    from src.llm.cost_tracking import CostTracker

    with patch.object(CostTracker, "__init__", return_value=None):
        mock_tracker = Mock(spec=CostTracker)
        mock_tracker.calculate_phase_costs.return_value = {
            "phase_name": "test_phase",
            "phase_index": 0,
            "generation_ids": ["test-gen-123"],
            "total_cost": 0.001,
            "currency": "USD",
            "total_tokens": 100,
            "total_prompt_tokens": 50,
            "total_completion_tokens": 50,
            "generation_count": 1,
        }
        mock_tracker.calculate_run_costs.return_value = {
            "total_cost": 0.001,
            "currency": "USD",
            "total_tokens": 100,
            "total_prompt_tokens": 50,
            "total_completion_tokens": 50,
            "total_generations": 1,
            "phase_costs": [mock_tracker.calculate_phase_costs.return_value],
        }

        with patch("src.cost_tracker.CostTracker", return_value=mock_tracker):
            yield mock_tracker
