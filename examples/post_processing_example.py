"""
Example demonstrating how to use the new post-processing functionality.

This example shows how to create different types of LLM phases with various
post-processor configurations to clean up and improve LLM-generated content.
"""

from pathlib import Path

from src.llm_model import LlmModel
from src.phase_factory import PhaseFactory
from src.post_processors import CustomPostProcessor


def create_custom_post_processor() -> CustomPostProcessor:
    """
    Create a custom post-processor that removes duplicate lines.

    Returns:
        CustomPostProcessor: A custom post-processor instance
    """

    def remove_duplicates(original_block: str, llm_block: str, **kwargs) -> str:
        """
        Remove duplicate lines from the LLM-generated block.

        Args:
            original_block (str): The original markdown block
            llm_block (str): The LLM-generated block
            **kwargs: Additional context

        Returns:
            str: The block with duplicates removed
        """
        lines = llm_block.split("\n")
        seen = set()
        unique_lines = []

        for line in lines:
            if line.strip() and line not in seen:
                seen.add(line)
                unique_lines.append(line)
            elif not line.strip():
                unique_lines.append(line)

        return "\n".join(unique_lines)

    return CustomPostProcessor(name="duplicate_remover", process_func=remove_duplicates)


def example_standard_phase_with_post_processing():
    """
    Example of creating a standard phase with built-in post-processors.
    """
    print("=== Standard Phase with Post-Processing ===")

    # Create model (you would use your actual model configuration)
    model = LlmModel(model_name="gpt-4", api_key="your-api-key")

    # Create phase with formatting and consistency post-processors
    phase = PhaseFactory.create_standard_phase(
        name="modernize_with_cleanup",
        input_file_path=Path("input.md"),
        output_file_path=Path("output.md"),
        original_file_path=Path("original.md"),
        system_prompt_path=Path("prompts/modernize_system.md"),
        user_prompt_path=Path("prompts/modernize_user.md"),
        book_name="Example Book",
        author_name="Example Author",
        model=model,
        temperature=0.2,
        post_processors=["formatting", "consistency"],
    )

    print(f"Created phase: {phase}")
    print(f"Post-processor chain: {phase.post_processor_chain}")
    print()


def example_annotation_phase_with_custom_post_processing():
    """
    Example of creating an annotation phase with custom post-processors.
    """
    print("=== Annotation Phase with Custom Post-Processing ===")

    # Create model (you would use your actual model configuration)
    model = LlmModel(model_name="gpt-4", api_key="your-api-key")

    # Create custom post-processor
    custom_processor = create_custom_post_processor()

    # Create introduction annotation phase with custom post-processor
    phase = PhaseFactory.create_introduction_annotation_phase(
        name="introduction_with_cleanup",
        input_file_path=Path("input.md"),
        output_file_path=Path("output.md"),
        original_file_path=Path("original.md"),
        system_prompt_path=Path("prompts/introduction_system.md"),
        user_prompt_path=Path("prompts/introduction_user.md"),
        book_name="Example Book",
        author_name="Example Author",
        model=model,
        temperature=0.2,
        custom_post_processors=[custom_processor],
    )

    print(f"Created phase: {phase}")
    print(f"Post-processor chain: {phase.post_processor_chain}")
    print()


def example_summary_phase_with_mixed_post_processing():
    """
    Example of creating a summary phase with both built-in and custom post-processors.
    """
    print("=== Summary Phase with Mixed Post-Processing ===")

    # Create model (you would use your actual model configuration)
    model = LlmModel(model_name="gpt-4", api_key="your-api-key")

    # Create custom post-processor
    custom_processor = create_custom_post_processor()

    # Create summary annotation phase with mixed post-processors
    phase = PhaseFactory.create_summary_annotation_phase(
        name="summary_with_cleanup",
        input_file_path=Path("input.md"),
        output_file_path=Path("output.md"),
        original_file_path=Path("original.md"),
        system_prompt_path=Path("prompts/summary_system.md"),
        user_prompt_path=Path("prompts/summary_user.md"),
        book_name="Example Book",
        author_name="Example Author",
        model=model,
        temperature=0.2,
        post_processors=["formatting", "validation"],
        custom_post_processors=[custom_processor],
    )

    print(f"Created phase: {phase}")
    print(f"Post-processor chain: {phase.post_processor_chain}")
    print()


def example_phase_without_post_processing():
    """
    Example of creating a phase without any post-processing (original behavior).
    """
    print("=== Phase Without Post-Processing ===")

    # Create model (you would use your actual model configuration)
    model = LlmModel(model_name="gpt-4", api_key="your-api-key")

    # Create phase without post-processors
    phase = PhaseFactory.create_standard_phase(
        name="modernize_no_cleanup",
        input_file_path=Path("input.md"),
        output_file_path=Path("output.md"),
        original_file_path=Path("original.md"),
        system_prompt_path=Path("prompts/modernize_system.md"),
        user_prompt_path=Path("prompts/modernize_user.md"),
        book_name="Example Book",
        author_name="Example Author",
        model=model,
        temperature=0.2,
        # No post_processors or custom_post_processors specified
    )

    print(f"Created phase: {phase}")
    print(f"Post-processor chain: {phase.post_processor_chain}")
    print()


def main():
    """
    Run all examples to demonstrate the post-processing functionality.
    """
    print("Post-Processing Examples")
    print("=" * 50)
    print()

    example_standard_phase_with_post_processing()
    example_annotation_phase_with_custom_post_processing()
    example_summary_phase_with_mixed_post_processing()
    example_phase_without_post_processing()

    print("All examples completed!")


if __name__ == "__main__":
    main()
