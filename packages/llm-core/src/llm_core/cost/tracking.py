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
import threading
from typing import Dict, List, Optional, Tuple

import requests  # type: ignore[import-untyped]
from loguru import logger
from pydantic import Field, model_validator

from llm_core.config import (
    BATCH_PROCESSING_DISCOUNT_RATE,
    GEMINI_FLASH_INPUT_PRICE_PER_1M,
    GEMINI_FLASH_LITE_INPUT_PRICE_PER_1M,
    GEMINI_FLASH_LITE_OUTPUT_PRICE_PER_1M,
    GEMINI_FLASH_OUTPUT_PRICE_PER_1M,
    GEMINI_PRO_INPUT_PRICE_PER_1M,
    GEMINI_PRO_OUTPUT_PRICE_PER_1M,
    OPENAI_GPT4O_INPUT_PRICE_PER_1M,
    OPENAI_GPT4O_MINI_INPUT_PRICE_PER_1M,
    OPENAI_GPT4O_MINI_OUTPUT_PRICE_PER_1M,
    OPENAI_GPT4O_OUTPUT_PRICE_PER_1M,
    OPENAI_GPT35_INPUT_PRICE_PER_1M,
    OPENAI_GPT35_OUTPUT_PRICE_PER_1M,
    OPENAI_O4_MINI_INPUT_PRICE_PER_1M,
    OPENAI_O4_MINI_OUTPUT_PRICE_PER_1M,
    OPENROUTER_MODELS_API_TIMEOUT,
    TOKENS_PER_MILLION,
    BaseConfig,
)
from llm_core.providers.types import Provider

# Initialize module-level logger
module_logger = logger


class GenerationStats(BaseConfig):
    """Data class for generation statistics from any LLM provider."""

    generation_id: str
    model: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost: float = 0.0
    currency: str = "USD"
    created_at: Optional[str] = None
    finish_reason: Optional[str] = None
    provider: Optional[Provider] = None
    is_estimated: bool = False
    estimation_method: Optional[str] = None

    @model_validator(mode="after")
    def validate_token_totals(self) -> "GenerationStats":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
        return self


class PhaseCosts(BaseConfig):
    """Data class for tracking costs per phase."""

    phase_name: str
    phase_index: int = Field(ge=0)
    generation_ids: List[str]
    total_prompt_tokens: int = Field(ge=0)
    total_completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    total_cost: float = 0.0
    currency: str = "USD"
    generation_count: int = Field(default=0, ge=0)
    estimated_count: int = Field(default=0, ge=0)
    actual_count: int = Field(default=0, ge=0)


class RunCosts(BaseConfig):
    """Data class for tracking total run costs."""

    total_phases: int = Field(ge=0)
    completed_phases: int = Field(ge=0)
    total_generations: int = Field(ge=0)
    total_prompt_tokens: int = Field(ge=0)
    total_completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    total_cost: float = 0.0
    phase_costs: List[PhaseCosts]
    currency: str = "USD"
    total_estimated_count: int = Field(default=0, ge=0)
    total_actual_count: int = Field(default=0, ge=0)


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

    # Fallback pricing per 1M tokens (USD). Prefer live pricing from OpenRouter models API.
    # These are imported from src.constants for maintainability.
    OPENAI_PRICING = {
        "o4-mini": {
            "input": OPENAI_O4_MINI_INPUT_PRICE_PER_1M,
            "output": OPENAI_O4_MINI_OUTPUT_PRICE_PER_1M,
        },
        "gpt-4o": {
            "input": OPENAI_GPT4O_INPUT_PRICE_PER_1M,
            "output": OPENAI_GPT4O_OUTPUT_PRICE_PER_1M,
        },
        "gpt-4o-mini": {
            "input": OPENAI_GPT4O_MINI_INPUT_PRICE_PER_1M,
            "output": OPENAI_GPT4O_MINI_OUTPUT_PRICE_PER_1M,
        },
        "gpt-3.5-turbo": {
            "input": OPENAI_GPT35_INPUT_PRICE_PER_1M,
            "output": OPENAI_GPT35_OUTPUT_PRICE_PER_1M,
        },
    }

    GEMINI_PRICING = {
        "gemini-2.5-flash": {
            "input": GEMINI_FLASH_INPUT_PRICE_PER_1M,
            "output": GEMINI_FLASH_OUTPUT_PRICE_PER_1M,
        },
        "gemini-2.5-pro": {
            "input": GEMINI_PRO_INPUT_PRICE_PER_1M,
            "output": GEMINI_PRO_OUTPUT_PRICE_PER_1M,
        },
        "gemini-2.5-flash-lite": {
            "input": GEMINI_FLASH_LITE_INPUT_PRICE_PER_1M,
            "output": GEMINI_FLASH_LITE_OUTPUT_PRICE_PER_1M,
        },
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
        # Cache of model pricing from OpenRouter models API.
        # Keys include OpenRouter ids (e.g., "openai/gpt-4o"), provider model names
        # (e.g., "gpt-4o"), and variants stripped of suffixes like ":free".
        self._model_pricing_index: Dict[str, Tuple[float, float]] = {}
        self._model_pricing_loaded: bool = False

        # Create a persistent session for API calls (minor optimization)
        self._session = requests.Session()
        self._session.headers.update(self.headers)

    def _load_openrouter_model_pricing(self) -> None:
        """
        Load pricing from OpenRouter models endpoint and build a fast lookup index.

        The endpoint returns per-token USD prices as strings under pricing.prompt and
        pricing.completion. We index by several keys to improve matching across providers:
        - Full OpenRouter id (e.g., "google/gemini-2.5-flash")
        - Suffix after provider (e.g., "gemini-2.5-flash")
        - Variant without any trailing ":..." tag (e.g., "kimi-k2" from "kimi-k2:free")
        """
        if self._model_pricing_loaded:
            return

        url = f"{self.base_url}/models" if self.base_url else "https://openrouter.ai/api/v1/models"
        try:
            # This endpoint is public; headers are optional, but include them if present.
            response = self._session.get(url, timeout=OPENROUTER_MODELS_API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])

            index: Dict[str, Tuple[float, float]] = {}
            for model in models:
                model_id = model.get("id")
                pricing = model.get("pricing", {}) or {}
                prompt_str = pricing.get("prompt")
                completion_str = pricing.get("completion")
                if not model_id or prompt_str is None or completion_str is None:
                    continue
                # Convert to floats (per-token USD)
                try:
                    prompt_price = float(prompt_str)
                    completion_price = float(completion_str)
                except (TypeError, ValueError):
                    continue

                keys: List[str] = []
                keys.append(model_id)

                # Add suffix after provider (e.g., "openai/gpt-4o" -> "gpt-4o")
                if "/" in model_id:
                    suffix = model_id.split("/", 1)[1]
                    keys.append(suffix)
                    # Strip any ":variant" suffix
                    if ":" in suffix:
                        base_variant = suffix.split(":", 1)[0]
                        keys.append(base_variant)

                # Also strip any :variant from full id and add as key
                if ":" in model_id:
                    base_id = model_id.split(":", 1)[0]
                    keys.append(base_id)

                for key in keys:
                    # Prefer first-seen price; don't overwrite to avoid oscillations
                    if key and key not in index:
                        index[key] = (prompt_price, completion_price)

            self._model_pricing_index = index
            self._model_pricing_loaded = True
            module_logger.info(f"Loaded OpenRouter pricing for {len(self._model_pricing_index)} model keys")
        except requests.exceptions.RequestException as e:
            module_logger.warning(f"Failed to load OpenRouter model pricing: {e}")
            self._model_pricing_loaded = False
        except (ValueError, json.JSONDecodeError) as e:
            module_logger.warning(f"Invalid response parsing OpenRouter models: {e}")
            self._model_pricing_loaded = False

    def _get_pricing_for_model(self, model: str) -> Optional[Tuple[float, float]]:
        """
        Get per-token USD pricing for the given model from OpenRouter models index.

        Returns a tuple of (prompt_price_per_token, completion_price_per_token) if found.
        """
        if not self._model_pricing_loaded:
            self._load_openrouter_model_pricing()

        if not model:
            return None

        # Try exact match
        if model in self._model_pricing_index:
            return self._model_pricing_index[model]

        # Try treating as OpenRouter id suffix (after provider)
        suffix = model.split("/", 1)[1] if "/" in model else model
        if suffix in self._model_pricing_index:
            return self._model_pricing_index[suffix]

        # Try removing any ":variant"
        if ":" in suffix and suffix.split(":", 1)[0] in self._model_pricing_index:
            return self._model_pricing_index[suffix.split(":", 1)[0]]

        # Try lowercased key as a last resort (index is case-sensitive but most ids are lowercase)
        lower = suffix.lower()
        if lower in self._model_pricing_index:
            return self._model_pricing_index[lower]

        return None

    def _detect_provider_from_generation_id(self, generation_id: str) -> Provider:
        """
        Detect the provider based on generation ID patterns.

        Args:
            generation_id: The generation ID to analyze

        Returns:
            Provider enum value
        """
        # OpenAI generation IDs:
        # - Legacy Chat Completions: "chatcmpl-..." / "cmpl-..."
        # - New Responses API: "resp_..."
        if (
            generation_id.startswith("chatcmpl-")
            or generation_id.startswith("cmpl-")
            or generation_id.startswith("resp_")
        ):
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
        # Prefer OpenRouter models pricing (per-token USD) if available
        pricing_tuple = self._get_pricing_for_model(model)
        if pricing_tuple is not None:
            prompt_price, completion_price = pricing_tuple
            total_cost = prompt_tokens * prompt_price + completion_tokens * completion_price
            return total_cost, "openrouter_models_pricing"

        # Fallback to static per-1M pricing heuristics
        pricing = None
        estimation_method = "exact_model_pricing_fallback"

        if model in self.OPENAI_PRICING:
            pricing = self.OPENAI_PRICING[model]
        else:
            # Check for more specific models first before generic patterns
            if "gpt-4o-mini" in model.lower() or "4o-mini" in model.lower():
                pricing = self.OPENAI_PRICING["gpt-4o-mini"]
                estimation_method = "similar_model_pricing_fallback"
            elif "o4-mini" in model.lower():
                pricing = self.OPENAI_PRICING["o4-mini"]
                estimation_method = "similar_model_pricing_fallback"
            elif "mini" in model.lower():
                # Default mini models to gpt-4o-mini (most common)
                pricing = self.OPENAI_PRICING["gpt-4o-mini"]
                estimation_method = "similar_model_pricing_fallback"
            elif "4o" in model or "gpt-4" in model:
                pricing = self.OPENAI_PRICING["gpt-4o"]
                estimation_method = "similar_model_pricing_fallback"
            else:
                pricing = self.OPENAI_PRICING["gpt-3.5-turbo"]
                estimation_method = "default_model_pricing_fallback"

        input_cost = (prompt_tokens / TOKENS_PER_MILLION) * pricing["input"]
        output_cost = (completion_tokens / TOKENS_PER_MILLION) * pricing["output"]
        total_cost = input_cost + output_cost
        return total_cost, estimation_method

    def _estimate_gemini_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int, is_batch: bool = False
    ) -> Tuple[float, str]:
        """
        Estimate cost for Gemini API calls based on token usage.

        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            is_batch: Whether this was batch processing (50% discount applies)

        Returns:
            Tuple of (estimated_cost, estimation_method)
        """
        # Prefer OpenRouter models pricing (per-token USD) if available
        pricing_tuple = self._get_pricing_for_model(model)
        if pricing_tuple is not None:
            prompt_price, completion_price = pricing_tuple
            total_cost = prompt_tokens * prompt_price + completion_tokens * completion_price
            # Apply batch discount if applicable
            if is_batch:
                total_cost *= BATCH_PROCESSING_DISCOUNT_RATE
                return total_cost, "openrouter_models_pricing_batch_discount"
            return total_cost, "openrouter_models_pricing"

        # Fallback to static per-1M pricing heuristics
        pricing = None
        estimation_method = "exact_model_pricing_fallback"

        if model in self.GEMINI_PRICING:
            pricing = self.GEMINI_PRICING[model]
        else:
            if "flash" in model.lower() and "lite" in model.lower():
                pricing = self.GEMINI_PRICING["gemini-2.5-flash-lite"]
                estimation_method = "similar_model_pricing_fallback"
            elif "flash" in model.lower():
                pricing = self.GEMINI_PRICING["gemini-2.5-flash"]
                estimation_method = "similar_model_pricing_fallback"
            elif "pro" in model.lower():
                pricing = self.GEMINI_PRICING["gemini-2.5-pro"]
                estimation_method = "similar_model_pricing_fallback"
            else:
                pricing = self.GEMINI_PRICING["gemini-2.5-flash"]
                estimation_method = "default_model_pricing_fallback"

        input_cost = (prompt_tokens / TOKENS_PER_MILLION) * pricing["input"]
        output_cost = (completion_tokens / TOKENS_PER_MILLION) * pricing["output"]
        total_cost = input_cost + output_cost

        # Apply batch discount if applicable
        if is_batch:
            total_cost *= BATCH_PROCESSING_DISCOUNT_RATE
            estimation_method += "_batch_discount"

        return total_cost, estimation_method

    def get_generation_stats(
        self,
        generation_id: str,
        model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        provider: Optional[Provider] = None,
        is_batch: bool = False,
    ) -> Optional[GenerationStats]:
        """
        Query or estimate generation statistics from any supported provider.

        Args:
            generation_id: The generation ID to query
            model: Model name (required for cost estimation with non-OpenRouter providers)
            prompt_tokens: Number of prompt tokens (for cost estimation)
            completion_tokens: Number of completion tokens (for cost estimation)
            provider: Provider to use for queries (if None, will detect from generation ID)
            is_batch: Whether this was batch processing (50% discount applies)

        Returns:
            GenerationStats object if successful, None otherwise
        """
        # Check cache first
        if generation_id in self.generation_stats_cache:
            return self.generation_stats_cache[generation_id]

        # Use provided provider or detect from generation ID
        if provider is None:
            provider = self._detect_provider_from_generation_id(generation_id)

        if provider == Provider.OPENROUTER:
            # Use OpenRouter API to get actual costs
            try:
                url = f"{self.base_url}/generation?id={generation_id}"
                response = self._session.get(url, timeout=30)
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

            except (requests.exceptions.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
                # If OpenRouter lookup fails but we have sufficient info, fall back to estimation
                module_logger.warning(
                    f"OpenRouter stats fetch failed for generation {generation_id}: {e}. "
                    "Attempting estimation if model/tokens were provided."
                )
                if model is not None and prompt_tokens is not None and completion_tokens is not None:
                    model_str: str = model
                    prompt_tokens_int: int = int(prompt_tokens)
                    completion_tokens_int: int = int(completion_tokens)

                    # Infer provider from model string for better estimation routing
                    lower_model = model_str.lower()
                    if "openai" in lower_model or lower_model.startswith("gpt") or lower_model.startswith("o4"):
                        estimated_cost, estimation_method = self._estimate_openai_cost(
                            model=model_str,
                            prompt_tokens=prompt_tokens_int,
                            completion_tokens=completion_tokens_int,
                        )
                        provider_for_estimation = Provider.OPENAI
                    elif "gemini" in lower_model or lower_model.startswith("google/"):
                        estimated_cost, estimation_method = self._estimate_gemini_cost(
                            model=model_str,
                            prompt_tokens=prompt_tokens_int,
                            completion_tokens=completion_tokens_int,
                            is_batch=is_batch,
                        )
                        provider_for_estimation = Provider.GEMINI
                    else:
                        # Default generic estimation using OpenAI pricing heuristics
                        estimated_cost, estimation_method = self._estimate_openai_cost(
                            model=model_str,
                            prompt_tokens=prompt_tokens_int,
                            completion_tokens=completion_tokens_int,
                        )
                        provider_for_estimation = Provider.OPENAI

                    stats = GenerationStats(
                        generation_id=generation_id,
                        model=model_str,
                        prompt_tokens=prompt_tokens_int,
                        completion_tokens=completion_tokens_int,
                        total_tokens=prompt_tokens_int + completion_tokens_int,
                        cost=estimated_cost,
                        currency="USD",
                        provider=provider_for_estimation,
                        is_estimated=True,
                        estimation_method=f"fallback_from_openrouter:{estimation_method}",
                    )
                    self.generation_stats_cache[generation_id] = stats
                    return stats
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
            model_name: str = model
            prompt_token_count: int = int(prompt_tokens)
            completion_token_count: int = int(completion_tokens)

            total_tokens = prompt_token_count + completion_token_count

            if provider == Provider.OPENAI:
                estimated_cost, estimation_method = self._estimate_openai_cost(
                    model=model_name,
                    prompt_tokens=prompt_token_count,
                    completion_tokens=completion_token_count,
                )
            elif provider == Provider.GEMINI:
                estimated_cost, estimation_method = self._estimate_gemini_cost(
                    model=model_name,
                    prompt_tokens=prompt_token_count,
                    completion_tokens=completion_token_count,
                    is_batch=is_batch,
                )
            else:
                module_logger.warning(f"Unsupported provider for cost estimation: {provider.value}")
                return None

            stats = GenerationStats(
                generation_id=generation_id,
                model=model_name,
                prompt_tokens=prompt_token_count,
                completion_tokens=completion_token_count,
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
        model_info: Optional[Dict[str, "ModelInfo"]] = None,
    ) -> PhaseCosts:
        """
        Calculate costs for a specific phase.

        Args:
            phase_name: Name of the phase
            phase_index: Index of the phase in the pipeline
            generation_ids: List of generation IDs for this phase
            model_info: Optional dictionary where each generation ID maps to a ModelInfo
                       dictionary containing keys "model" (Optional[str]), "prompt_tokens" (Optional[int]),
                       "completion_tokens" (Optional[int]), and "is_batch" (bool)

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
            gen_model_info = model_info.get(gen_id) if model_info else None

            # Extract provider with proper type handling
            provider_value = gen_model_info.provider if gen_model_info else None
            provider: Optional[Provider] = provider_value if isinstance(provider_value, Provider) else None

            stats = self.get_generation_stats(
                generation_id=gen_id,
                model=gen_model_info.model if gen_model_info else None,
                prompt_tokens=gen_model_info.prompt_tokens if gen_model_info else None,
                completion_tokens=gen_model_info.completion_tokens if gen_model_info else None,
                provider=provider,
                is_batch=gen_model_info.is_batch if gen_model_info else False,
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


# ============================================================================
# Cost Tracking Wrapper
# ============================================================================


class ModelInfo(BaseConfig):
    """Model information stored with generation IDs."""

    model: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    is_batch: bool = False
    provider: Provider | None = None


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
        # Import here to avoid circular dependency
        from llm_core.config import settings

        if api_key is None:
            api_key = settings.get_api_key("openrouter")

        if api_key:
            self.cost_tracker: CostTracker | None = CostTracker(api_key=api_key)
            self.enabled = True
            module_logger.info("Cost tracking enabled")
        else:
            self.cost_tracker = None
            self.enabled = False
            module_logger.warning("No API key found, cost tracking disabled")

        # Store generation IDs by phase
        self.phase_generations: Dict[str, List[str]] = {}
        # Store model information for cost estimation
        self.model_info: Dict[str, ModelInfo] = {}
        # Thread lock for synchronizing access to shared data structures

        self._lock = threading.Lock()

    def add_generation_id(
        self,
        phase_name: str,
        generation_id: str,
        model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> None:
        """
        Add a generation ID for a specific phase with optional model information for cost estimation.

        Args:
            phase_name: Name of the phase
            generation_id: Generation ID from the API call
            model: Model name (required for non-OpenRouter providers)
            prompt_tokens: Number of prompt tokens (for cost estimation)
            completion_tokens: Number of completion tokens (for cost estimation)
        """
        if not self.enabled:
            return

        with self._lock:
            if phase_name not in self.phase_generations:
                self.phase_generations[phase_name] = []

            self.phase_generations[phase_name].append(generation_id)

            # Store model information if provided
            if model or prompt_tokens is not None or completion_tokens is not None:
                self.model_info[generation_id] = ModelInfo(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

        module_logger.debug(f"Added generation ID {generation_id} for phase {phase_name}")

    def set_generation_model_info(
        self,
        generation_id: str,
        model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        is_batch: bool = False,
    ) -> None:
        """
        Register or update model/token information for a generation ID.

        This is useful for providers (e.g., OpenAI) where token usage is known
        at call time but phase association is handled elsewhere.

        Args:
            generation_id: Unique identifier for this generation
            model: Model name used
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            is_batch: Whether this was batch processing (applies 50% discount)
        """
        if not self.enabled:
            return

        with self._lock:
            self.model_info[generation_id] = ModelInfo(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                is_batch=is_batch,
            )
        module_logger.debug(
            f"Registered model info for generation {generation_id}: model={model}, "
            f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}"
        )

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

            # Create snapshots of shared data structures to avoid holding lock during calculations
            with self._lock:
                phase_generations_snapshot = {phase: ids.copy() for phase, ids in self.phase_generations.items()}
                model_info_snapshot = self.model_info.copy()

            # Calculate costs for each phase
            phase_costs = []
            for i, phase_name in enumerate(phase_names):
                generation_ids = phase_generations_snapshot.get(phase_name, [])
                if generation_ids:
                    phase_cost = self.cost_tracker.calculate_phase_costs(
                        phase_name=phase_name,
                        phase_index=i,
                        generation_ids=generation_ids,
                        model_info=model_info_snapshot,
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

        except (KeyError, ValueError, AttributeError, TypeError) as e:
            # Cost calculation errors are logged but shouldn't break the pipeline
            module_logger.error(f"Failed to calculate run costs: {e}")
            module_logger.exception("Cost calculation error details")
            return None

    def get_phase_generation_count(self, phase_name: str) -> int:
        """
        Get the number of generations for a specific phase.

        Args:
            phase_name: Name of the phase

        Returns:
            Number of generations for the phase
        """
        with self._lock:
            return len(self.phase_generations.get(phase_name, []))

    def get_total_generation_count(self) -> int:
        """
        Get the total number of generations across all phases.

        Returns:
            Total number of generations
        """
        with self._lock:
            return sum(len(generations) for generations in self.phase_generations.values())

    def clear_generations(self) -> None:
        """Clear all stored generation IDs and model information."""
        with self._lock:
            self.phase_generations.clear()
            self.model_info.clear()
        module_logger.debug("Cleared all generation IDs and model information")


# Global instance for easy access
_cost_tracking_wrapper: Optional[CostTrackingWrapper] = None
_cost_tracking_lock = threading.Lock()


def get_cost_tracking_wrapper() -> Optional[CostTrackingWrapper]:
    """
    Get the global cost tracking wrapper instance (thread-safe).

    Returns:
        CostTrackingWrapper instance if available, None otherwise
    """
    global _cost_tracking_wrapper
    # First check without lock for performance
    if _cost_tracking_wrapper is None:
        # Acquire lock for initialization
        with _cost_tracking_lock:
            # Double-check inside lock to prevent race condition
            if _cost_tracking_wrapper is None:
                _cost_tracking_wrapper = CostTrackingWrapper()
    return _cost_tracking_wrapper


def add_generation_id(
    phase_name: str,
    generation_id: str,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> None:
    """
    Add a generation ID for a specific phase with optional model information.

    Args:
        phase_name: Name of the phase
        generation_id: Generation ID from the API call
        model: Model name (required for non-OpenRouter providers)
        prompt_tokens: Number of prompt tokens (for cost estimation)
        completion_tokens: Number of completion tokens (for cost estimation)
    """
    wrapper = get_cost_tracking_wrapper()
    if wrapper:
        wrapper.add_generation_id(phase_name, generation_id, model, prompt_tokens, completion_tokens)


def register_generation_model_info(
    generation_id: str,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    is_batch: bool = False,
) -> None:
    """
    Public helper to register or update model/token info for a generation ID.

    Args:
        generation_id: Unique identifier for this generation
        model: Model name used
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        is_batch: Whether this was batch processing (applies 50% discount)
    """
    wrapper = get_cost_tracking_wrapper()
    if wrapper:
        wrapper.set_generation_model_info(
            generation_id=generation_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            is_batch=is_batch,
        )


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


def add_llm_generation(
    phase_name: str,
    generation_id: str,
    model_config,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> None:
    """
    Add a generation ID from an LlmModel call with automatic model name extraction.

    Args:
        phase_name: Name of the phase
        generation_id: Generation ID from the LlmModel call
        model_config: ModelConfig object from LlmModel
        prompt_tokens: Number of prompt tokens (for cost estimation)
        completion_tokens: Number of completion tokens (for cost estimation)
    """
    # Extract model name from ModelConfig
    model_name = getattr(model_config, "provider_model_name", None) or getattr(model_config, "model_id", None)

    add_generation_id(
        phase_name=phase_name,
        generation_id=generation_id,
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
