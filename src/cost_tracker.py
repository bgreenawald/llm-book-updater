"""
Cost tracking module for multiple LLM providers (OpenRouter, OpenAI, Gemini).

This module provides functionality to track and calculate costs for LLM API calls:
- OpenRouter: Uses actual cost data from the generation stats API endpoint
- OpenAI: Estimates costs based on token usage and current pricing
- Gemini: Estimates costs based on token usage and current pricing

Usage:
    # For OpenRouter (automatic cost retrieval)
    tracker = CostTracker(api_key="your_openrouter_key")
    stats = tracker.get_generation_stats(generation_id)

    # For OpenAI/Gemini (cost estimation)
    stats = tracker.get_generation_stats(
        generation_id="chatcmpl-xyz",  # OpenAI format
        model="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50
    )

Note: Cost estimates for OpenAI and Gemini are based on publicly available
pricing and may not reflect actual charges, especially for enterprise pricing.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import requests

from src.logging_config import setup_logging

# Initialize module-level logger
module_logger = setup_logging(log_name="cost_tracker")


class Provider(Enum):
    """Enumeration of supported LLM providers for cost tracking."""

    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class GenerationStats:
    """Data class for generation statistics from any LLM provider."""

    generation_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    currency: str = "USD"
    created_at: Optional[str] = None
    finish_reason: Optional[str] = None
    provider: Optional[Provider] = None
    is_estimated: bool = False
    estimation_method: Optional[str] = None


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
    estimated_count: int = 0
    actual_count: int = 0


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
    total_estimated_count: int = 0
    total_actual_count: int = 0


class CostTracker:
    """
    Tracks costs for multiple LLM providers (OpenRouter, OpenAI, Gemini) across phases and runs.

    This class provides functionality to:
    - Query generation statistics from OpenRouter
    - Estimate costs for OpenAI and Gemini based on token usage
    - Calculate costs per phase
    - Aggregate costs for the entire run
    - Log cost information
    """

    # Cost per 1M tokens (as of January 2025) - approximate pricing
    # These prices are estimates based on public pricing and may vary
    OPENAI_PRICING = {
        "o4-mini": {"input": 0.15, "output": 0.60},  # $0.15/$0.60 per 1M tokens
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    GEMINI_PRICING = {
        "gemini-2.5-flash": {"input": 0.075, "output": 0.30},  # $0.075/$0.30 per 1M tokens
        "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    }

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        """
        Initialize the cost tracker.

        Args:
            api_key: OpenRouter API key (still needed for OpenRouter calls)
            base_url: OpenRouter API base URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.generation_stats_cache: Dict[str, GenerationStats] = {}

    def _detect_provider_from_generation_id(self, generation_id: str) -> Provider:
        """
        Detect the provider based on generation ID patterns.

        Args:
            generation_id: The generation ID to analyze

        Returns:
            Provider enum value
        """
        # OpenAI generation IDs typically start with "chatcmpl-"
        if generation_id.startswith("chatcmpl-"):
            return Provider.OPENAI

        # Gemini generation IDs are custom generated with timestamp
        if generation_id.startswith("gemini_"):
            return Provider.GEMINI

        # Default to OpenRouter for other patterns
        return Provider.OPENROUTER

    def _estimate_openai_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Tuple[float, str]:
        """
        Estimate cost for OpenAI API calls based on token usage.

        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Tuple of (estimated_cost, estimation_method)
        """
        # Try to find pricing for the exact model
        pricing = None
        estimation_method = "exact_model_pricing"

        if model in self.OPENAI_PRICING:
            pricing = self.OPENAI_PRICING[model]
        else:
            # Fallback to similar model pricing
            if "4o-mini" in model or "mini" in model.lower():
                pricing = self.OPENAI_PRICING["o4-mini"]
                estimation_method = "similar_model_pricing"
            elif "4o" in model or "gpt-4" in model:
                pricing = self.OPENAI_PRICING["gpt-4o"]
                estimation_method = "similar_model_pricing"
            else:
                # Default to GPT-3.5 pricing
                pricing = self.OPENAI_PRICING["gpt-3.5-turbo"]
                estimation_method = "default_model_pricing"

        # Calculate cost: (tokens / 1M) * price_per_1M
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return total_cost, estimation_method

    def _estimate_gemini_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Tuple[float, str]:
        """
        Estimate cost for Gemini API calls based on token usage.

        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Tuple of (estimated_cost, estimation_method)
        """
        # Try to find pricing for the exact model
        pricing = None
        estimation_method = "exact_model_pricing"

        if model in self.GEMINI_PRICING:
            pricing = self.GEMINI_PRICING[model]
        else:
            # Fallback to similar model pricing
            if "flash" in model.lower() and "lite" in model.lower():
                pricing = self.GEMINI_PRICING["gemini-2.5-flash-lite"]
                estimation_method = "similar_model_pricing"
            elif "flash" in model.lower():
                pricing = self.GEMINI_PRICING["gemini-2.5-flash"]
                estimation_method = "similar_model_pricing"
            elif "pro" in model.lower():
                pricing = self.GEMINI_PRICING["gemini-2.5-pro"]
                estimation_method = "similar_model_pricing"
            else:
                # Default to Flash pricing
                pricing = self.GEMINI_PRICING["gemini-2.5-flash"]
                estimation_method = "default_model_pricing"

        # Calculate cost: (tokens / 1M) * price_per_1M
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return total_cost, estimation_method

    def get_generation_stats(
        self,
        generation_id: str,
        model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> Optional[GenerationStats]:
        """
        Query or estimate generation statistics from any supported provider.

        Args:
            generation_id: The generation ID to query
            model: Model name (required for cost estimation with non-OpenRouter providers)
            prompt_tokens: Number of prompt tokens (for cost estimation)
            completion_tokens: Number of completion tokens (for cost estimation)

        Returns:
            GenerationStats object if successful, None otherwise
        """
        # Check cache first
        if generation_id in self.generation_stats_cache:
            return self.generation_stats_cache[generation_id]

        # Detect provider from generation ID
        provider = self._detect_provider_from_generation_id(generation_id)

        if provider == Provider.OPENROUTER:
            # Use OpenRouter API to get actual costs
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
                    provider=Provider.OPENROUTER,
                    is_estimated=False,
                    estimation_method=None,
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

        else:
            # For OpenAI and Gemini, estimate costs based on token usage
            # Validate and narrow optional types before arithmetic/estimation
            if model is None or prompt_tokens is None or completion_tokens is None:
                module_logger.warning(
                    f"Missing required parameters for cost estimation of {provider.value} generation {generation_id}"
                )
                return None

            # Local non-optional copies for type checkers
            model_str: str = model
            prompt_tokens_int: int = int(prompt_tokens)
            completion_tokens_int: int = int(completion_tokens)

            total_tokens = prompt_tokens_int + completion_tokens_int

            if provider == Provider.OPENAI:
                estimated_cost, estimation_method = self._estimate_openai_cost(
                    model=model_str,
                    prompt_tokens=prompt_tokens_int,
                    completion_tokens=completion_tokens_int,
                )
            elif provider == Provider.GEMINI:
                estimated_cost, estimation_method = self._estimate_gemini_cost(
                    model=model_str,
                    prompt_tokens=prompt_tokens_int,
                    completion_tokens=completion_tokens_int,
                )
            else:
                module_logger.warning(f"Unsupported provider for cost estimation: {provider.value}")
                return None

            stats = GenerationStats(
                generation_id=generation_id,
                model=model_str,
                prompt_tokens=prompt_tokens_int,
                completion_tokens=completion_tokens_int,
                total_tokens=total_tokens,
                cost=estimated_cost,
                currency="USD",
                provider=provider,
                is_estimated=True,
                estimation_method=estimation_method,
            )

            # Cache the result
            self.generation_stats_cache[generation_id] = stats

            module_logger.debug(
                f"Estimated stats for {provider.value} generation {generation_id}: "
                f"{stats.total_tokens} tokens, ${stats.cost:.6f} (method: {estimation_method})"
            )
            return stats

    def calculate_phase_costs(
        self,
        phase_name: str,
        phase_index: int,
        generation_ids: List[str],
        model_info: Optional[Dict[str, Dict]] = None,
    ) -> PhaseCosts:
        """
        Calculate costs for a specific phase.

        Args:
            phase_name: Name of the phase
            phase_index: Index of the phase in the pipeline
            generation_ids: List of generation IDs for this phase
            model_info: Optional dict mapping generation_id to model/token info

        Returns:
            PhaseCosts object with aggregated statistics
        """
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        currency = "USD"
        valid_generations = 0
        estimated_count = 0
        actual_count = 0

        for gen_id in generation_ids:
            # Get model info for this generation if available
            gen_model_info = model_info.get(gen_id, {}) if model_info else {}

            stats = self.get_generation_stats(
                generation_id=gen_id,
                model=gen_model_info.get("model"),
                prompt_tokens=gen_model_info.get("prompt_tokens"),
                completion_tokens=gen_model_info.get("completion_tokens"),
            )

            if stats:
                total_prompt_tokens += stats.prompt_tokens
                total_completion_tokens += stats.completion_tokens
                total_cost += stats.cost
                currency = stats.currency  # Use currency from last valid generation
                valid_generations += 1

                if stats.is_estimated:
                    estimated_count += 1
                else:
                    actual_count += 1

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
            estimated_count=estimated_count,
            actual_count=actual_count,
        )

        module_logger.info(
            f"Phase {phase_name} costs: {total_tokens} tokens "
            f"({total_prompt_tokens} prompt, {total_completion_tokens} completion), "
            f"${total_cost:.6f} {currency}, {valid_generations} generations "
            f"({actual_count} actual, {estimated_count} estimated)"
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
        total_estimated_count = sum(phase.estimated_count for phase in phase_costs)
        total_actual_count = sum(phase.actual_count for phase in phase_costs)

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
            total_estimated_count=total_estimated_count,
            total_actual_count=total_actual_count,
        )

        module_logger.success(
            f"Total run costs: {total_tokens} tokens "
            f"({total_prompt_tokens} prompt, {total_completion_tokens} completion), "
            f"${total_cost:.6f} {currency}, {total_generations} generations across {len(phase_costs)} phases "
            f"({total_actual_count} actual, {total_estimated_count} estimated)"
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
                module_logger.info(
                    f"  Generations: {phase.generation_count} "
                    f"({phase.actual_count} actual, {phase.estimated_count} estimated)"
                )
                module_logger.info("")

        module_logger.info("=" * 80)
        module_logger.info("TOTAL RUN SUMMARY")
        module_logger.info("=" * 80)
        module_logger.info(f"Total Phases: {run_costs.total_phases}")
        module_logger.info(f"Completed Phases: {run_costs.completed_phases}")
        module_logger.info(
            f"Total Generations: {run_costs.total_generations:,} "
            f"({run_costs.total_actual_count:,} actual, {run_costs.total_estimated_count:,} estimated)"
        )
        module_logger.info(
            f"Total Tokens: {run_costs.total_tokens:,} "
            f"({run_costs.total_prompt_tokens:,} prompt, {run_costs.total_completion_tokens:,} completion)"
        )
        module_logger.info(f"Total Cost: ${run_costs.total_cost:.6f} {run_costs.currency}")
        if run_costs.total_estimated_count > 0:
            module_logger.info(
                f"Note: {run_costs.total_estimated_count:,} costs are estimated "
                f"based on token usage and current pricing"
            )
        module_logger.info("=" * 80)
