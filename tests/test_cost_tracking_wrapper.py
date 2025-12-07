"""
Tests for cost tracking wrapper functionality.

These tests verify that the cost tracking wrapper works correctly,
including thread-safe singleton initialization.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from src import cost_tracking_wrapper
from src.cost_tracking_wrapper import (
    CostTrackingWrapper,
    get_cost_tracking_wrapper,
)


@pytest.fixture(autouse=True)
def reset_global_wrapper():
    """Reset the global wrapper instance and lock before each test."""
    cost_tracking_wrapper._cost_tracking_wrapper = None
    cost_tracking_wrapper._cost_tracking_lock = threading.Lock()
    yield
    cost_tracking_wrapper._cost_tracking_wrapper = None
    cost_tracking_wrapper._cost_tracking_lock = threading.Lock()


class TestCostTrackingWrapper:
    """Tests for the CostTrackingWrapper class."""

    def test_wrapper_initialization_with_api_key(self):
        """Test that wrapper initializes correctly with an API key."""
        wrapper = CostTrackingWrapper(api_key="test-api-key")
        assert wrapper.enabled is True
        assert wrapper.cost_tracker is not None

    def test_wrapper_initialization_without_api_key(self):
        """Test that wrapper initializes correctly without an API key."""
        # Temporarily remove the API key from environment for this test
        with patch.dict(os.environ, {}, clear=True):
            wrapper = CostTrackingWrapper(api_key=None)
            assert wrapper.enabled is False
            assert wrapper.cost_tracker is None

    def test_add_generation_id(self):
        """Test adding a generation ID to the wrapper."""
        wrapper = CostTrackingWrapper(api_key="test-api-key")
        wrapper.add_generation_id(
            phase_name="test_phase",
            generation_id="gen-123",
            model="test/model",
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert wrapper.get_phase_generation_count("test_phase") == 1
        assert wrapper.get_total_generation_count() == 1

    def test_get_phase_generation_count(self):
        """Test getting generation count for a phase."""
        wrapper = CostTrackingWrapper(api_key="test-api-key")
        wrapper.add_generation_id("phase1", "gen-1")
        wrapper.add_generation_id("phase1", "gen-2")
        wrapper.add_generation_id("phase2", "gen-3")

        assert wrapper.get_phase_generation_count("phase1") == 2
        assert wrapper.get_phase_generation_count("phase2") == 1
        assert wrapper.get_phase_generation_count("nonexistent") == 0

    def test_get_total_generation_count(self):
        """Test getting total generation count across all phases."""
        wrapper = CostTrackingWrapper(api_key="test-api-key")
        wrapper.add_generation_id("phase1", "gen-1")
        wrapper.add_generation_id("phase1", "gen-2")
        wrapper.add_generation_id("phase2", "gen-3")

        assert wrapper.get_total_generation_count() == 3

    def test_clear_generations(self):
        """Test clearing all generations."""
        wrapper = CostTrackingWrapper(api_key="test-api-key")
        wrapper.add_generation_id("phase1", "gen-1")
        wrapper.add_generation_id("phase2", "gen-2")

        wrapper.clear_generations()

        assert wrapper.get_total_generation_count() == 0
        assert wrapper.get_phase_generation_count("phase1") == 0
        assert wrapper.get_phase_generation_count("phase2") == 0


class TestGlobalCostTrackingWrapper:
    """Tests for the global cost tracking wrapper singleton."""

    def test_get_cost_tracking_wrapper_returns_instance(self):
        """Test that get_cost_tracking_wrapper returns an instance."""
        wrapper = get_cost_tracking_wrapper()
        assert wrapper is not None
        assert isinstance(wrapper, CostTrackingWrapper)

    def test_get_cost_tracking_wrapper_returns_same_instance(self):
        """Test that multiple calls return the same instance."""
        wrapper1 = get_cost_tracking_wrapper()
        wrapper2 = get_cost_tracking_wrapper()

        assert wrapper1 is wrapper2
        assert id(wrapper1) == id(wrapper2)

    def test_concurrent_wrapper_access(self):
        """Test that concurrent access returns the same singleton instance."""
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_cost_tracking_wrapper) for _ in range(100)]
            results = [f.result() for f in futures]

        # All should return the same instance
        assert len(set(id(r) for r in results if r is not None)) == 1

        # Verify all results are the same object
        first_result = results[0]
        assert all(r is first_result for r in results)
