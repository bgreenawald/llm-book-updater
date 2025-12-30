"""Cost tracking for LLM API usage.

This module provides a lightweight cost tracking system that can be extended
for detailed cost analysis.
"""

from typing import Optional


def register_generation_model_info(
    generation_id: str,
    model: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    is_batch: bool = False,
) -> None:
    """Register model information for a generation.

    This is a placeholder for cost tracking that can be extended
    to track actual costs across generations.

    Args:
        generation_id: Unique identifier for the generation
        model: Name of the model used
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens used
        is_batch: Whether this was a batch API call (50% discount)
    """
    # Placeholder - can be extended to track costs
    pass
