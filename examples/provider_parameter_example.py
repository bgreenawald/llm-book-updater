"""
Example script demonstrating how to use OpenRouter's provider parameter for routing control.

The provider parameter allows you to control routing preferences when using OpenRouter,
including specifying provider order, filtering by data collection policies, and more.

See: https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request
"""

import sys
from pathlib import Path
from typing import List

# Add project root to path to allow importing from src
sys.path.append(str(Path(__file__).parent.parent))

from src.api.config import PhaseConfig, PhaseType, RunConfig
from src.api.provider import Provider
from src.core.pipeline import run_pipeline
from src.models import GROK_3_MINI, ModelConfig


def main() -> None:
    """Run the pipeline with OpenRouter provider routing preferences."""

    # Example 1: Prioritize specific providers in order
    provider_with_order = PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        model=GROK_3_MINI,
        llm_kwargs={
            "provider": {
                "order": ["openai", "anthropic"],  # Try OpenAI first, then Anthropic
                "allow_fallbacks": True,  # Allow fallbacks if primary unavailable
            }
        },
    )

    # Example 2: Only use providers that don't collect data
    provider_privacy_focused = PhaseConfig(
        phase_type=PhaseType.EDIT,
        model=ModelConfig(provider=Provider.OPENROUTER, model_id="anthropic/claude-sonnet-4"),
        llm_kwargs={
            "provider": {
                "data_collection": "deny",  # Only use providers that don't collect data
            }
        },
    )

    # Example 3: Sort by price (cheapest first)
    provider_cost_optimized = PhaseConfig(
        phase_type=PhaseType.FINAL,
        model=GROK_3_MINI,
        llm_kwargs={
            "provider": {
                "sort": "price",  # Sort by price (cheapest first)
                "allow_fallbacks": True,
            }
        },
    )

    # Example 4: Set maximum price limits
    provider_budget_limited = PhaseConfig(
        phase_type=PhaseType.SUMMARY,
        model=ModelConfig(provider=Provider.OPENROUTER, model_id="anthropic/claude-sonnet-4"),
        llm_kwargs={
            "provider": {
                "max_price": {
                    "prompt": 0.001,  # Max price per million prompt tokens
                    "completion": 0.005,  # Max price per million completion tokens
                }
            }
        },
    )

    # Example 5: Exclude specific providers
    provider_with_exclusions = PhaseConfig(
        phase_type=PhaseType.ANNOTATE,
        model=GROK_3_MINI,
        llm_kwargs={
            "provider": {
                "ignore": ["provider-a", "provider-b"],  # Exclude specific providers
            }
        },
    )

    # Define phases using different provider configurations
    run_phases: List[PhaseConfig] = [
        provider_with_order,
        provider_privacy_focused,
        provider_cost_optimized,
        provider_budget_limited,
        provider_with_exclusions,
    ]

    # Create run configuration
    config = RunConfig(
        book_id="provider_example",
        book_name="Provider Example",
        author_name="Example Author",
        input_file=Path("books/on_liberty/input_small.md"),
        output_dir=Path("books/on_liberty/output"),
        original_file=Path("books/on_liberty/input_small.md"),
        phases=run_phases,
    )

    # Run the pipeline
    print("Running pipeline with provider routing preferences...")
    print("\nPhase configurations:")
    for i, phase in enumerate(run_phases, 1):
        provider = phase.llm_kwargs.get("provider", "default") if phase.llm_kwargs else "default"
        print(f"{i}. {phase.phase_type.name}: {provider}")

    run_pipeline(config)
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
