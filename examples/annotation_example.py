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

from src.config import PhaseConfig, PhaseType, RunConfig
from src.llm_model import LlmModel
from src.pipeline import Pipeline


def main():
    """Run the annotation example."""

    # Configuration
    book_name = "Example Book"
    author_name = "Example Author"
    temperature = 0.2
    max_workers = 4

    # File paths
    input_file = Path("input.md")
    original_file = Path("original.md")
    output_dir = Path("output")

    # Create dummy files for the example to run
    input_file.touch()
    original_file.touch()

    # Prompt paths
    system_prompts_dir = Path("prompts")
    introduction_prompt = system_prompts_dir / "introduction_system.md"
    summary_prompt = system_prompts_dir / "summary_system.md"
    standard_prompt = system_prompts_dir / "edit_system.md"  # Example standard prompt

    # User prompt paths (optional)
    user_prompts_dir = Path("prompts")
    user_prompt = user_prompts_dir / "edit_user.md"  # Example user prompt

    # Initialize the LLM model
    model = LlmModel(
        model="gpt-4",
        temperature=temperature,
    )

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
                model_type=model.model_id,
                temperature=temperature,
                max_workers=max_workers,
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
                model_type=model.model_id,
                temperature=temperature,
                max_workers=max_workers,
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
                model_type=model.model_id,
                temperature=temperature,
                max_workers=max_workers,
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
                model_type=model.model_id,
                temperature=temperature,
                max_workers=max_workers,
                system_prompt_path=introduction_prompt,
                user_prompt_path=user_prompt,
            ),
            PhaseConfig(
                phase_type=PhaseType.SUMMARY,
                model_type=model.model_id,
                temperature=temperature,
                max_workers=max_workers,
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
