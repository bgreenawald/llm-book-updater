"""
Cost tracking module for OpenRouter API usage.

This module provides functionality to track and calculate costs for LLM API calls
using OpenRouter's generation stats endpoint.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

from src.logging_config import setup_logging

# Initialize module-level logger
module_logger = setup_logging(log_name="cost_tracker")


@dataclass
class GenerationStats:
    """Data class for generation statistics from OpenRouter."""

    generation_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    currency: str = "USD"
    created_at: Optional[str] = None
    finish_reason: Optional[str] = None


@dataclass
class PhaseCosts:
    """Data class for tracking costs per phase."""

    phase_name: str
    phase_index: int
    generation_ids: List[str]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost: float
    currency: str = "USD"
    generation_count: int = 0


@dataclass
class RunCosts:
    """Data class for tracking total run costs."""

    total_phases: int
    completed_phases: int
    total_generations: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost: float
    phase_costs: List[PhaseCosts]
    currency: str = "USD"


class CostTracker:
    """
    Tracks costs for OpenRouter API usage across phases and runs.

    This class provides functionality to:
    - Query generation statistics from OpenRouter
    - Calculate costs per phase
    - Aggregate costs for the entire run
    - Log cost information
    """

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        """
        Initialize the cost tracker.

        Args:
            api_key: OpenRouter API key
            base_url: OpenRouter API base URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.generation_stats_cache: Dict[str, GenerationStats] = {}

    def get_generation_stats(self, generation_id: str) -> Optional[GenerationStats]:
        """
        Query generation statistics from OpenRouter.

        Args:
            generation_id: The generation ID to query

        Returns:
            GenerationStats object if successful, None otherwise
        """
        # Check cache first
        if generation_id in self.generation_stats_cache:
            return self.generation_stats_cache[generation_id]

        try:
            url = f"{self.base_url}/generation?id={generation_id}"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            data = response.json().get("data", {})

            # Extract stats from response using correct field names
            stats = GenerationStats(
                generation_id=generation_id,
                model=data.get("model", "unknown"),
                prompt_tokens=data.get("tokens_prompt", 0),
                completion_tokens=data.get("tokens_completion", 0),
                total_tokens=data.get("tokens_prompt", 0) + data.get("tokens_completion", 0),
                cost=data.get("total_cost", 0.0),
                currency="USD",  # OpenRouter uses USD
                created_at=data.get("created_at"),
                finish_reason=data.get("finish_reason"),
            )

            # Cache the result
            self.generation_stats_cache[generation_id] = stats

            module_logger.debug(
                f"Retrieved stats for generation {generation_id}: {stats.total_tokens} tokens, ${stats.cost:.6f}"
            )
            return stats

        except requests.exceptions.RequestException as e:
            module_logger.warning(f"Failed to retrieve stats for generation {generation_id}: {e}")
            return None
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            module_logger.warning(f"Invalid response format for generation {generation_id}: {e}")
            return None

    def calculate_phase_costs(self, phase_name: str, phase_index: int, generation_ids: List[str]) -> PhaseCosts:
        """
        Calculate costs for a specific phase.

        Args:
            phase_name: Name of the phase
            phase_index: Index of the phase in the pipeline
            generation_ids: List of generation IDs for this phase

        Returns:
            PhaseCosts object with aggregated statistics
        """
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        currency = "USD"
        valid_generations = 0

        for gen_id in generation_ids:
            stats = self.get_generation_stats(gen_id)
            if stats:
                total_prompt_tokens += stats.prompt_tokens
                total_completion_tokens += stats.completion_tokens
                total_cost += stats.cost
                currency = stats.currency  # Use currency from last valid generation
                valid_generations += 1

        total_tokens = total_prompt_tokens + total_completion_tokens

        phase_costs = PhaseCosts(
            phase_name=phase_name,
            phase_index=phase_index,
            generation_ids=generation_ids,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            total_cost=total_cost,
            currency=currency,
            generation_count=valid_generations,
        )

        module_logger.info(
            f"Phase {phase_name} costs: {total_tokens} tokens "
            f"({total_prompt_tokens} prompt, {total_completion_tokens} completion), "
            f"${total_cost:.6f} {currency}, {valid_generations} generations"
        )

        return phase_costs

    def calculate_run_costs(self, phase_costs: List[PhaseCosts]) -> RunCosts:
        """
        Calculate total costs for the entire run.

        Args:
            phase_costs: List of PhaseCosts objects

        Returns:
            RunCosts object with aggregated statistics
        """
        total_generations = sum(phase.generation_count for phase in phase_costs)
        total_prompt_tokens = sum(phase.total_prompt_tokens for phase in phase_costs)
        total_completion_tokens = sum(phase.total_completion_tokens for phase in phase_costs)
        total_tokens = sum(phase.total_tokens for phase in phase_costs)
        total_cost = sum(phase.total_cost for phase in phase_costs)
        currency = phase_costs[0].currency if phase_costs else "USD"

        run_costs = RunCosts(
            total_phases=len(phase_costs),
            completed_phases=len([p for p in phase_costs if p.generation_count > 0]),
            total_generations=total_generations,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            total_cost=total_cost,
            currency=currency,
            phase_costs=phase_costs,
        )

        module_logger.success(
            f"Total run costs: {total_tokens} tokens "
            f"({total_prompt_tokens} prompt, {total_completion_tokens} completion), "
            f"${total_cost:.6f} {currency}, {total_generations} generations across {len(phase_costs)} phases"
        )

        return run_costs

    def log_detailed_costs(self, run_costs: RunCosts) -> None:
        """
        Log detailed cost breakdown.

        Args:
            run_costs: RunCosts object with complete cost information
        """
        module_logger.info("=" * 80)
        module_logger.info("DETAILED COST BREAKDOWN")
        module_logger.info("=" * 80)

        for phase in run_costs.phase_costs:
            if phase.generation_count > 0:
                module_logger.info(f"Phase {phase.phase_index + 1}: {phase.phase_name}")
                module_logger.info(
                    f"  Tokens: {phase.total_tokens:,} "
                    f"({phase.total_prompt_tokens:,} prompt, {phase.total_completion_tokens:,} completion)"
                )
                module_logger.info(f"  Cost: ${phase.total_cost:.6f} {phase.currency}")
                module_logger.info(f"  Generations: {phase.generation_count}")
                module_logger.info("")

        module_logger.info("=" * 80)
        module_logger.info("TOTAL RUN SUMMARY")
        module_logger.info("=" * 80)
        module_logger.info(f"Total Phases: {run_costs.total_phases}")
        module_logger.info(f"Completed Phases: {run_costs.completed_phases}")
        module_logger.info(f"Total Generations: {run_costs.total_generations:,}")
        module_logger.info(
            f"Total Tokens: {run_costs.total_tokens:,} "
            f"({run_costs.total_prompt_tokens:,} prompt, {run_costs.total_completion_tokens:,} completion)"
        )
        module_logger.info(f"Total Cost: ${run_costs.total_cost:.6f} {run_costs.currency}")
        module_logger.info("=" * 80)
