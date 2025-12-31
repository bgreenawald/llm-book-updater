"""
Example script demonstrating how sections containing only special tags are skipped.

This example shows how the pipeline now skips processing sections that contain
only special tags like {preface} and {license} after removing blank lines.
"""

from pathlib import Path

from book_updater import PhaseConfig, PhaseType, RunConfig


def main():
    """Demonstrate how sections with only special tags are skipped."""

    print("=== Special Tags Skip Example ===")
    print()
    print("This example demonstrates how the pipeline now skips processing")
    print("sections that contain only special tags (like {preface}, {license})")
    print("after removing blank lines.")
    print()

    # Example 1: Default tags_to_preserve
    print("=== Example 1: Default tags_to_preserve ===")
    config1 = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=[PhaseConfig(phase_type=PhaseType.MODERNIZE)],
        # Uses default: tags_to_preserve=["{preface}", "{license}"]
    )
    print(f"Config 1 tags_to_preserve: {config1.tags_to_preserve}")
    print("This will skip sections containing only {preface} or {license} tags")
    print()

    # Example 2: Custom tags_to_preserve
    print("=== Example 2: Custom tags_to_preserve ===")
    config2 = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=[PhaseConfig(phase_type=PhaseType.MODERNIZE)],
        tags_to_preserve=["{preface}", "{license}", "{dedication}", "{acknowledgments}"],
    )
    print(f"Config 2 tags_to_preserve: {config2.tags_to_preserve}")
    print("This will skip sections containing only {preface}, {license}, {dedication}, or {acknowledgments} tags")
    print()

    # Example 3: No tags to preserve
    print("=== Example 3: No tags to preserve ===")
    config3 = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=[PhaseConfig(phase_type=PhaseType.MODERNIZE)],
        tags_to_preserve=[],  # Don't preserve any tags
    )
    print(f"Config 3 tags_to_preserve: {config3.tags_to_preserve}")
    print("This will process all sections, including those with special tags")
    print()

    print("=== How It Works ===")
    print()
    print("The pipeline now checks if a section contains only special tags:")
    print("1. Extracts the body content from the markdown block")
    print("2. Removes all blank lines")
    print("3. Checks if all remaining lines are special tags")
    print("4. If yes, skips LLM processing and returns the block as-is")
    print()
    print("Examples of sections that would be skipped:")
    print("- A section containing only: {preface}")
    print("- A section containing only: {license}")
    print("- A section containing: {preface}\\n{license}")
    print("- A section containing: {preface}\\n\\n{license}  (blank lines removed)")
    print()
    print("Examples of sections that would be processed:")
    print("- A section containing: Some actual content")
    print("- A section containing: {preface}\\nSome actual content")
    print("- A section containing: Some content\\n{license}")
    print()

    print("=== Benefits ===")
    print("- Prevents LLM from generating content for structural elements")
    print("- Saves API calls and processing time")
    print("- Preserves document structure and formatting")
    print("- Configurable per run based on document requirements")
    print("- Works with all phase types (MODERNIZE, EDIT, ANNOTATE, etc.)")

    # Note: In a real scenario, you would run the pipeline with one of these configs
    # run_pipeline(config1)  # or config2, config3, etc.


if __name__ == "__main__":
    main()
