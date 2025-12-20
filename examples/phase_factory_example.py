"""
Example demonstrating the refactored PhaseFactory usage with unified post-processors.

This example shows how the PhaseFactory methods now accept a single PhaseConfig
instance with a unified post_processors list that can contain both built-in
processor names (strings) and custom processor instances.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PhaseConfig, PhaseType, PostProcessorType
from src.post_processors import EnsureBlankLineProcessor, RemoveXmlTagsProcessor


def main():
    """Demonstrate the refactored PhaseFactory usage with unified post-processors."""

    # Create a PhaseConfig for a standard phase with mixed post-processors
    standard_config = PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        name="modernize_content",
        book_name="The Great Gatsby",
        author_name="F. Scott Fitzgerald",
        input_file_path=Path("input/modernize.md"),
        output_file_path=Path("output/modernize.md"),
        original_file_path=Path("original/gatsby.md"),
        # Unified post-processors list: mix of strings and instances
        post_processors=[
            PostProcessorType.NO_NEW_HEADERS,  # Built-in processor by enum
            EnsureBlankLineProcessor(),  # Custom processor instance
        ],
    )

    # Create a PhaseConfig for an introduction annotation phase
    intro_config = PhaseConfig(
        phase_type=PhaseType.INTRODUCTION,
        name="add_introductions",
        book_name="The Great Gatsby",
        author_name="F. Scott Fitzgerald",
        input_file_path=Path("input/intro.md"),
        output_file_path=Path("output/intro.md"),
        original_file_path=Path("original/gatsby.md"),
        # Only built-in processors
        post_processors=[PostProcessorType.REMOVE_XML_TAGS],
    )

    # Create a PhaseConfig for a summary annotation phase
    summary_config = PhaseConfig(
        phase_type=PhaseType.SUMMARY,
        name="add_summaries",
        book_name="The Great Gatsby",
        author_name="F. Scott Fitzgerald",
        input_file_path=Path("input/summary.md"),
        output_file_path=Path("output/summary.md"),
        original_file_path=Path("original/gatsby.md"),
        # Mix of built-in and custom processors
        post_processors=[
            PostProcessorType.REVERT_REMOVED_BLOCK_LINES,
            RemoveXmlTagsProcessor(),  # Custom instance
        ],
    )

    print("=== PhaseFactory with Unified Post-Processors Example ===")
    print()

    print("1. Standard Phase Configuration:")
    print(f"   - Phase Type: {standard_config.phase_type}")
    print(f"   - Name: {standard_config.name}")
    print(f"   - Post Processors: {standard_config.post_processors}")
    print()

    print("2. Introduction Annotation Phase Configuration:")
    print(f"   - Phase Type: {intro_config.phase_type}")
    print(f"   - Name: {intro_config.name}")
    print(f"   - Post Processors: {intro_config.post_processors}")
    print()

    print("3. Summary Annotation Phase Configuration:")
    print(f"   - Phase Type: {summary_config.phase_type}")
    print(f"   - Name: {summary_config.name}")
    print(f"   - Post Processors: {summary_config.post_processors}")
    print()

    print("Benefits of the unified post-processor approach:")
    print("- Single post_processors list instead of two separate lists")
    print("- Can mix built-in processors (strings) and custom processors (instances)")
    print("- Simpler configuration and less confusion")
    print("- More flexible and intuitive API")
    print("- Easier to maintain and extend")

    # Note: The actual phase creation would require a model instance
    # which is not available in this example environment
    print()
    print("Note: Actual phase creation would require a model instance.")
    print("The configurations above demonstrate the new unified API structure.")


if __name__ == "__main__":
    main()
