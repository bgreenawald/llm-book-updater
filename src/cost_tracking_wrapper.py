"""
Cost tracking wrapper for existing LLM functionality.

This module provides a wrapper that can be used to add cost tracking
to existing LLM calls without modifying the core classes.
"""

import os
from typing import List, Optional

from src.cost_tracker import CostTracker, RunCosts
from src.logging_config import setup_logging

# Initialize module-level logger
module_logger = setup_logging(log_name="cost_tracking_wrapper")


class CostTrackingWrapper:
    """
    Wrapper class that adds cost tracking to existing LLM functionality.

    This class can be used to track costs without modifying existing code.
    It maintains a list of generation IDs and provides methods to calculate
    and log costs at the end of processing.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the cost tracking wrapper.

        Args:
            api_key: OpenRouter API key. If None, will try to get from environment.
        """
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")

        if api_key:
            self.cost_tracker = CostTracker(api_key=api_key)
            self.enabled = True
            module_logger.info("Cost tracking enabled")
        else:
            self.cost_tracker = None
            self.enabled = False
            module_logger.warning("No API key found, cost tracking disabled")

        # Store generation IDs by phase
        self.phase_generations: dict = {}

    def add_generation_id(self, phase_name: str, generation_id: str) -> None:
        """
        Add a generation ID for a specific phase.

        Args:
            phase_name: Name of the phase
            generation_id: Generation ID from the API call
        """
        if not self.enabled:
            return

        if phase_name not in self.phase_generations:
            self.phase_generations[phase_name] = []

        self.phase_generations[phase_name].append(generation_id)
        module_logger.debug(f"Added generation ID {generation_id} for phase {phase_name}")

    def calculate_and_log_costs(self, phase_names: List[str]) -> Optional[RunCosts]:
        """
        Calculate and log costs for all tracked phases.

        Args:
            phase_names: List of phase names in order

        Returns:
            RunCosts object if successful, None if cost tracking is disabled
        """
        if not self.enabled or not self.cost_tracker:
            module_logger.info("Cost tracking is disabled")
            return None

        try:
            module_logger.info("Calculating run costs...")

            # Calculate costs for each phase
            phase_costs = []
            for i, phase_name in enumerate(phase_names):
                generation_ids = self.phase_generations.get(phase_name, [])
                if generation_ids:
                    phase_cost = self.cost_tracker.calculate_phase_costs(
                        phase_name=phase_name,
                        phase_index=i,
                        generation_ids=generation_ids,
                    )
                    phase_costs.append(phase_cost)

            # Calculate total run costs
            if phase_costs:
                run_costs = self.cost_tracker.calculate_run_costs(phase_costs=phase_costs)
                self.cost_tracker.log_detailed_costs(run_costs=run_costs)
                return run_costs
            else:
                module_logger.info("No API calls were made, no costs to calculate")
                return None

        except Exception as e:
            module_logger.error(f"Failed to calculate run costs: {e}")
            return None

    def get_phase_generation_count(self, phase_name: str) -> int:
        """
        Get the number of generations for a specific phase.

        Args:
            phase_name: Name of the phase

        Returns:
            Number of generations for the phase
        """
        return len(self.phase_generations.get(phase_name, []))

    def get_total_generation_count(self) -> int:
        """
        Get the total number of generations across all phases.

        Returns:
            Total number of generations
        """
        return sum(len(generations) for generations in self.phase_generations.values())

    def clear_generations(self) -> None:
        """Clear all stored generation IDs."""
        self.phase_generations.clear()
        module_logger.debug("Cleared all generation IDs")


# Global instance for easy access
_cost_tracking_wrapper: Optional[CostTrackingWrapper] = None


def get_cost_tracking_wrapper() -> Optional[CostTrackingWrapper]:
    """
    Get the global cost tracking wrapper instance.

    Returns:
        CostTrackingWrapper instance if available, None otherwise
    """
    global _cost_tracking_wrapper
    if _cost_tracking_wrapper is None:
        _cost_tracking_wrapper = CostTrackingWrapper()
    return _cost_tracking_wrapper


def add_generation_id(phase_name: str, generation_id: str) -> None:
    """
    Add a generation ID for a specific phase.

    Args:
        phase_name: Name of the phase
        generation_id: Generation ID from the API call
    """
    wrapper = get_cost_tracking_wrapper()
    if wrapper:
        wrapper.add_generation_id(phase_name, generation_id)


def calculate_and_log_costs(phase_names: List[str]) -> Optional[RunCosts]:
    """
    Calculate and log costs for all tracked phases.

    Args:
        phase_names: List of phase names in order

    Returns:
        RunCosts object if successful, None if cost tracking is disabled
    """
    wrapper = get_cost_tracking_wrapper()
    if wrapper:
        return wrapper.calculate_and_log_costs(phase_names)
    return None
