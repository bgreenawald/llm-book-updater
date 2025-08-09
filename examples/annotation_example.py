#!/usr/bin/env python3
"""
Example script demonstrating how to use the new annotation phase classes.

This example shows how to:
1. Use IntroductionAnnotationPhase to add introductions to each section
2. Use SummaryAnnotationPhase to add summaries to each section
3. Use StandardLlmPhase for traditional content replacement
4. Use user prompts with title parameters
"""

from pathlib import Path

from src.common.provider import Provider
from src.config import PhaseConfig, PhaseType, RunConfig
from src.llm_model import ModelConfig
from src.pipeline import Pipeline


def main():
    """Run the annotation example."""

    # Configuration
    book_name = "The Great Gatsby"
    author_name = "F. Scott Fitzgerald"
    input_file = Path("examples/annotation_input.md")
    output_dir = Path("examples/annotation_output")
    temperature = 0.2

    # File paths
    original_file = Path("original.md")

    # Prompt paths
    system_prompts_dir = Path("prompts")
    introduction_prompt = system_prompts_dir / "introduction_system.md"
    summary_prompt = system_prompts_dir / "summary_system.md"
    standard_prompt = system_prompts_dir / "edit_system.md"  # Example standard prompt

    # User prompt paths (optional)
    user_prompts_dir = Path("prompts")
    user_prompt = user_prompts_dir / "edit_user.md"  # Example user prompt

    # Define the model configuration
    model = ModelConfig(Provider.OPENAI, "gpt-4")

    # Example 1: Add introduction annotations to each section
    print("Running introduction annotation phase...")
    intro_config = RunConfig(
        book_name=book_name,
        author_name=author_name,
        input_file=input_file,
        output_dir=output_dir,
        original_file=original_file,
        phases=[
            PhaseConfig(
                phase_type=PhaseType.INTRODUCTION,
                model=model,
                temperature=temperature,
                system_prompt_path=introduction_prompt,
                user_prompt_path=user_prompt,
            )
        ],
    )
    intro_pipeline = Pipeline(config=intro_config)
    intro_pipeline.run()
    print("Introduction annotations completed!")

    # Example 2: Add summary annotations to each section
    print("Running summary annotation phase...")
    summary_config = RunConfig(
        book_name=book_name,
        author_name=author_name,
        input_file=input_file,
        output_dir=output_dir,
        original_file=original_file,
        phases=[
            PhaseConfig(
                phase_type=PhaseType.SUMMARY,
                model=model,
                temperature=temperature,
                system_prompt_path=summary_prompt,
                user_prompt_path=user_prompt,
            )
        ],
    )
    summary_pipeline = Pipeline(config=summary_config)
    summary_pipeline.run()
    print("Summary annotations completed!")

    # Example 3: Use the standard phase for content replacement
    print("Running standard content replacement phase...")
    standard_config = RunConfig(
        book_name=book_name,
        author_name=author_name,
        input_file=input_file,
        output_dir=output_dir,
        original_file=original_file,
        phases=[
            PhaseConfig(
                phase_type=PhaseType.EDIT,
                model=model,
                temperature=temperature,
                system_prompt_path=standard_prompt,
                user_prompt_path=user_prompt,
            )
        ],
    )
    standard_pipeline = Pipeline(config=standard_config)
    standard_pipeline.run()
    print("Content replacement completed!")

    # Example 4: Chain phases together
    print("Running chained phases...")
    chained_config = RunConfig(
        book_name=book_name,
        author_name=author_name,
        input_file=input_file,
        output_dir=output_dir,
        original_file=original_file,
        phases=[
            PhaseConfig(
                phase_type=PhaseType.INTRODUCTION,
                model=model,
                temperature=temperature,
                system_prompt_path=introduction_prompt,
                user_prompt_path=user_prompt,
            ),
            PhaseConfig(
                phase_type=PhaseType.SUMMARY,
                model=model,
                temperature=temperature,
                system_prompt_path=summary_prompt,
                user_prompt_path=user_prompt,
            ),
        ],
    )
    chained_pipeline = Pipeline(config=chained_config)
    chained_pipeline.run()

    print("Chained phases completed!")
    print(f"Results saved in {output_dir}/")


if __name__ == "__main__":
    # Run the main example
    main()
