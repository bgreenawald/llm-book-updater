"""
Example demonstrating the default post-processor configuration for different phase types.

This example shows how each phase type automatically gets its default post-processors
when no explicit post_processors are specified in the PhaseConfig.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PhaseConfig, PhaseType, PostProcessorType
from src.phase_factory import PhaseFactory


def main():
    """Demonstrate default post-processor configurations for different phase types."""

    print("=== Default Post-Processor Configuration Example ===")
    print()

    # Create PhaseConfig instances for each phase type without specifying post_processors
    phase_configs = {
        PhaseType.MODERNIZE: PhaseConfig(
            phase_type=PhaseType.MODERNIZE,
            name="modernize_content",
            book_name="The Great Gatsby",
            author_name="F. Scott Fitzgerald",
            input_file_path=Path("input/modernize.md"),
            output_file_path=Path("output/modernize.md"),
            original_file_path=Path("original/gatsby.md"),
            temperature=0.3,
        ),
        PhaseType.EDIT: PhaseConfig(
            phase_type=PhaseType.EDIT,
            name="edit_content",
            book_name="The Great Gatsby",
            author_name="F. Scott Fitzgerald",
            input_file_path=Path("input/edit.md"),
            output_file_path=Path("output/edit.md"),
            original_file_path=Path("original/gatsby.md"),
            temperature=0.2,
        ),
        PhaseType.FINAL: PhaseConfig(
            phase_type=PhaseType.FINAL,
            name="final_content",
            book_name="The Great Gatsby",
            author_name="F. Scott Fitzgerald",
            input_file_path=Path("input/final.md"),
            output_file_path=Path("output/final.md"),
            original_file_path=Path("original/gatsby.md"),
            temperature=0.1,
        ),
        PhaseType.INTRODUCTION: PhaseConfig(
            phase_type=PhaseType.INTRODUCTION,
            name="add_introductions",
            book_name="The Great Gatsby",
            author_name="F. Scott Fitzgerald",
            input_file_path=Path("input/intro.md"),
            output_file_path=Path("output/intro.md"),
            original_file_path=Path("original/gatsby.md"),
            temperature=0.2,
        ),
        PhaseType.SUMMARY: PhaseConfig(
            phase_type=PhaseType.SUMMARY,
            name="add_summaries",
            book_name="The Great Gatsby",
            author_name="F. Scott Fitzgerald",
            input_file_path=Path("input/summary.md"),
            output_file_path=Path("output/summary.md"),
            original_file_path=Path("original/gatsby.md"),
            temperature=0.1,
        ),
        PhaseType.ANNOTATE: PhaseConfig(
            phase_type=PhaseType.ANNOTATE,
            name="add_annotations",
            book_name="The Great Gatsby",
            author_name="F. Scott Fitzgerald",
            input_file_path=Path("input/annotate.md"),
            output_file_path=Path("output/annotate.md"),
            original_file_path=Path("original/gatsby.md"),
            temperature=0.2,
        ),
    }

    # Display the default post-processor configurations
    print("Default Post-Processor Configurations by Phase Type:")
    print("=" * 60)
    print()

    for phase_type, default_processors in PhaseFactory.DEFAULT_POST_PROCESSORS.items():
        print(f"{phase_type.name}:")
        for i, processor in enumerate(default_processors, 1):
            print(f"  {i}. {processor.name}")
        print()

    # Demonstrate creating phases with default post-processors
    print("Creating phases with default post-processors:")
    print("=" * 50)
    print()

    for phase_type, config in phase_configs.items():
        print(f"Creating {phase_type.name} phase...")

        # Create the appropriate phase type
        if phase_type in [PhaseType.MODERNIZE, PhaseType.EDIT, PhaseType.FINAL, PhaseType.ANNOTATE]:
            phase = PhaseFactory.create_standard_phase(config)
        elif phase_type == PhaseType.INTRODUCTION:
            phase = PhaseFactory.create_introduction_annotation_phase(config)
        elif phase_type == PhaseType.SUMMARY:
            phase = PhaseFactory.create_summary_annotation_phase(config)
        else:
            print(f"  Unknown phase type: {phase_type}")
            continue

        # Display the post-processor chain
        if phase.post_processor_chain:
            print(f"  Post-processor chain: {phase.post_processor_chain}")
            print(f"  Number of processors: {len(phase.post_processor_chain)}")
        else:
            print("  No post-processors configured")
        print()

    # Demonstrate different ways to specify post-processors
    print("Different Ways to Specify Post-Processors:")
    print("=" * 45)
    print()

    # Example 1: Using string names (legacy approach)
    string_config = PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        name="modernize_strings",
        book_name="The Great Gatsby",
        author_name="F. Scott Fitzgerald",
        input_file_path=Path("input/modernize.md"),
        output_file_path=Path("output/modernize.md"),
        original_file_path=Path("original/gatsby.md"),
        temperature=0.3,
        # Using string names
        post_processors=["remove_xml_tags", "ensure_blank_line"],
    )

    string_phase = PhaseFactory.create_standard_phase(string_config)
    print("1. Using string names:")
    print(f"   Post-processor chain: {string_phase.post_processor_chain}")
    print()

    # Example 2: Using PostProcessorType enum (type-safe approach)
    enum_config = PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        name="modernize_enums",
        book_name="The Great Gatsby",
        author_name="F. Scott Fitzgerald",
        input_file_path=Path("input/modernize.md"),
        output_file_path=Path("output/modernize.md"),
        original_file_path=Path("original/gatsby.md"),
        temperature=0.3,
        # Using PostProcessorType enum values
        post_processors=[
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.ENSURE_BLANK_LINE,
        ],
    )

    enum_phase = PhaseFactory.create_standard_phase(enum_config)
    print("2. Using PostProcessorType enum:")
    print(f"   Post-processor chain: {enum_phase.post_processor_chain}")
    print()

    # Example 3: Mixing different approaches
    from src.post_processors import RemoveTrailingWhitespaceProcessor

    mixed_config = PhaseConfig(
        phase_type=PhaseType.ANNOTATE,
        name="annotate_mixed",
        book_name="The Great Gatsby",
        author_name="F. Scott Fitzgerald",
        input_file_path=Path("input/annotate.md"),
        output_file_path=Path("output/annotate.md"),
        original_file_path=Path("original/gatsby.md"),
        temperature=0.2,
        # Mixing strings, enums, and instances
        post_processors=[
            "revert_removed_block_lines",  # String
            PostProcessorType.ORDER_QUOTE_ANNOTATION,  # Enum
            RemoveTrailingWhitespaceProcessor(),  # Instance
            PostProcessorType.ENSURE_BLANK_LINE,  # Enum
        ],
    )

    mixed_phase = PhaseFactory.create_standard_phase(mixed_config)
    print("3. Mixing different approaches:")
    print(f"   Post-processor chain: {mixed_phase.post_processor_chain}")
    print()

    print("Benefits of PostProcessorType enum:")
    print("- Type safety and IDE autocompletion")
    print("- Prevents typos in processor names")
    print("- Clear documentation of available processors")
    print("- Consistent with PhaseType enum pattern")
    print()
    print("Benefits of default post-processor configuration:")
    print("- No need to manually specify post-processors for common use cases")
    print("- Consistent post-processing across similar phase types")
    print("- Easy to override when custom processing is needed")
    print("- Reduces configuration boilerplate")


if __name__ == "__main__":
    main()
