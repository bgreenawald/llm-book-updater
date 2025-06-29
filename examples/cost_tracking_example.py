#!/usr/bin/env python3
"""
Example script demonstrating cost tracking functionality.

This script shows how cost tracking is now automatically integrated
into the pipeline and will log costs at the end of the run.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import PhaseConfig, PhaseType, RunConfig
from llm_model import LlmModel
from pipeline import run_pipeline


def create_simple_config() -> RunConfig:
    """
    Create a simple configuration for testing cost tracking.

    Returns:
        RunConfig: Configuration with a single modernize phase
    """
    # Create a simple input file for testing
    input_file = Path("examples/test_input.md")
    input_file.parent.mkdir(exist_ok=True)

    with open(input_file, "w") as f:
        f.write("""# Test Chapter

This is a test chapter to demonstrate cost tracking functionality.

## Section 1

This is the first section with some content that will be processed by the LLM.

## Section 2

This is the second section with more content for testing.

## Section 3

This is the final section to complete our test content.
""")

    # Create output directory
    output_dir = Path("examples/cost_tracking_output")
    output_dir.mkdir(exist_ok=True)

    # Create model
    model = LlmModel.create(
        model="openai/gpt-3.5-turbo",  # Use a cost-effective model for testing
        temperature=0.2,
    )

    # Create phase configuration
    phase_config = PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        name="modernize",
        model=model,
        temperature=0.2,
        enabled=True,
        system_prompt_path=Path("prompts/modernize_system.md"),
        user_prompt_path=Path("prompts/modernize_user.md"),
        post_processors=None,
        reasoning=None,
    )

    # Create run configuration
    config = RunConfig(
        book_name="Cost Tracking Test Book",
        author_name="Test Author",
        input_file=input_file,
        original_file=input_file,
        output_dir=output_dir,
        phases=[phase_config],
        length_reduction=None,
    )

    return config


def main():
    """Main function to run the cost tracking example."""
    print("Cost Tracking Example")
    print("=" * 50)

    # Check if API key is available
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key to test cost tracking:")
        print("export OPENROUTER_API_KEY='your-api-key-here'")
        return

    print("✓ OpenRouter API key found")
    print("✓ Cost tracking is automatically integrated into the pipeline")
    print("✓ Costs will be logged at the end of the run")
    print()

    try:
        # Create configuration
        config = create_simple_config()

        print("Configuration created:")
        print(f"  Book: {config.book_name} by {config.author_name}")
        print(f"  Input file: {config.input_file}")
        print(f"  Output directory: {config.output_dir}")
        print(f"  Phases: {len(config.phases)}")
        print()

        # Run pipeline - cost tracking is now automatic!
        print("Running pipeline with automatic cost tracking...")
        print("-" * 50)

        run_pipeline(config)

        print("-" * 50)
        print("Pipeline completed!")
        print()
        print("Cost information has been automatically logged above.")
        print("Check the output directory for:")
        print("  - Processed markdown files")
        print("  - Pipeline metadata (pipeline_metadata_*.json)")
        print()
        print("The cost tracking is now fully integrated into the pipeline!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
