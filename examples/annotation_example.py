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

from src.llm_model import LlmModel
from src.llm_phase import (
    IntroductionAnnotationPhase,
    StandardLlmPhase,
    SummaryAnnotationPhase,
)


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

    # Prompt paths
    system_prompts_dir = Path("prompts")
    introduction_prompt = system_prompts_dir / "introduction_annotation.md"
    summary_prompt = system_prompts_dir / "summary_annotation.md"
    standard_prompt = system_prompts_dir / "edit_system.md"  # Example standard prompt

    # User prompt paths (optional)
    user_prompts_dir = Path("user_prompts")
    user_prompt = user_prompts_dir / "custom_user_prompt.md"  # Example user prompt

    # Initialize the LLM model
    model = LlmModel(
        model_name="gpt-4",
        api_key="your-api-key-here",  # Replace with actual API key
        base_url="https://api.openai.com/v1",
    )

    # Example 1: Add introduction annotations to each section
    print("Running introduction annotation phase...")
    intro_phase = IntroductionAnnotationPhase(
        name="introduction_annotation",
        input_file_path=input_file,
        output_file_path=output_dir / "with_introductions.md",
        original_file_path=original_file,
        system_prompt_path=introduction_prompt,
        user_prompt_path=user_prompt,
        book_name=book_name,
        author_name=author_name,
        model=model,
        temperature=temperature,
        max_workers=max_workers,
    )

    intro_phase.run()
    print("Introduction annotations completed!")

    # Example 2: Add summary annotations to each section
    print("Running summary annotation phase...")
    summary_phase = SummaryAnnotationPhase(
        name="summary_annotation",
        input_file_path=input_file,
        output_file_path=output_dir / "with_summaries.md",
        original_file_path=original_file,
        system_prompt_path=summary_prompt,
        user_prompt_path=user_prompt,
        book_name=book_name,
        author_name=author_name,
        model=model,
        temperature=temperature,
        max_workers=max_workers,
    )

    summary_phase.run()
    print("Summary annotations completed!")

    # Example 3: Use the standard phase for content replacement
    print("Running standard content replacement phase...")
    standard_phase = StandardLlmPhase(
        name="content_replacement",
        input_file_path=input_file,
        output_file_path=output_dir / "replaced_content.md",
        original_file_path=original_file,
        system_prompt_path=standard_prompt,
        user_prompt_path=user_prompt,
        book_name=book_name,
        author_name=author_name,
        model=model,
        temperature=temperature,
        max_workers=max_workers,
    )

    standard_phase.run()
    print("Content replacement completed!")

    # Example 4: Use phases with custom user prompts that include titles
    print("Running phases with custom user prompts...")

    # Standard phase with user prompt that uses titles
    standard_with_user_prompt = StandardLlmPhase(
        name="content_replacement_with_user_prompt",
        input_file_path=input_file,
        output_file_path=output_dir / "replaced_with_user_prompt.md",
        original_file_path=original_file,
        system_prompt_path=standard_prompt,
        user_prompt_path=user_prompt,  # User prompt with title variables
        book_name=book_name,
        author_name=author_name,
        model=model,
        temperature=temperature,
        max_workers=max_workers,
    )

    standard_with_user_prompt.run()
    print("Content replacement with user prompt completed!")

    # Example 5: Chain phases together
    print("Running chained phases...")

    # First, add introductions
    intro_phase = IntroductionAnnotationPhase(
        name="chained_intro",
        input_file_path=input_file,
        output_file_path=output_dir / "temp_with_intros.md",
        original_file_path=original_file,
        system_prompt_path=introduction_prompt,
        user_prompt_path=user_prompt,
        book_name=book_name,
        author_name=author_name,
        model=model,
        temperature=temperature,
        max_workers=max_workers,
    )
    intro_phase.run()

    # Then, add summaries to the result
    summary_phase = SummaryAnnotationPhase(
        name="chained_summary",
        input_file_path=output_dir / "temp_with_intros.md",
        output_file_path=output_dir / "with_intros_and_summaries.md",
        original_file_path=original_file,
        system_prompt_path=summary_prompt,
        user_prompt_path=user_prompt,
        book_name=book_name,
        author_name=author_name,
        model=model,
        temperature=temperature,
        max_workers=max_workers,
    )
    summary_phase.run()

    print("Chained phases completed!")
    print(f"Results saved in {output_dir}/")


def create_example_user_prompt():
    """Create an example user prompt file that demonstrates title usage."""
    user_prompt_content = """# Section Analysis Request

Please analyze the following section and provide your response.

**Section Title:** {transformed_title}
**Original Title:** {original_title}

**Current Content:**
{transformed_passage}

**Original Content:**
{original_passage}

Please process this section according to the system instructions.
"""

    user_prompts_dir = Path("user_prompts")
    user_prompts_dir.mkdir(exist_ok=True)

    with open(user_prompts_dir / "custom_user_prompt.md", "w") as f:
        f.write(user_prompt_content)

    print("Created example user prompt file: user_prompts/custom_user_prompt.md")


if __name__ == "__main__":
    # Create example user prompt file
    create_example_user_prompt()

    # Run the main example
    main()
