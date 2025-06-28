#!/usr/bin/env python3
"""
Example demonstrating different ways to use the length_reduction parameter.

This example shows how to configure the run with different length reduction settings.
"""

import sys
from pathlib import Path

# Add project root to path to allow importing from src
sys.path.append(str(Path(__file__).parent.parent))

from src.config import PhaseConfig, PhaseType, RunConfig
from src.llm_model import OPENAI_04_MINI


def example_single_percentage():
    """Example with a single percentage value."""
    print("Example 1: Single percentage (40%)")

    run_phases = [
        PhaseConfig(
            phase_type=PhaseType.EDIT,
            model_type=OPENAI_04_MINI,
            temperature=0.2,
        ),
    ]

    config = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=run_phases,
        length_reduction=40,  # Single percentage value
    )

    print(f"Length reduction: {config.length_reduction}%")
    # run_pipeline(config)  # Uncomment to run


def example_range():
    """Example with a range of percentages."""
    print("Example 2: Range (25-60%)")

    run_phases = [
        PhaseConfig(
            phase_type=PhaseType.EDIT,
            model_type=OPENAI_04_MINI,
            temperature=0.2,
        ),
    ]

    config = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=run_phases,
        length_reduction=(25, 60),  # Range of percentages
    )

    print(f"Length reduction: {config.length_reduction}")
    # run_pipeline(config)  # Uncomment to run


def example_multiple_phases():
    """Example with multiple phases using run-level length reduction."""
    print("Example 3: Multiple phases with run-level length reduction")

    run_phases = [
        PhaseConfig(
            phase_type=PhaseType.MODERNIZE,
            model_type=OPENAI_04_MINI,
            temperature=0.3,
        ),
        PhaseConfig(
            phase_type=PhaseType.EDIT,
            model_type=OPENAI_04_MINI,
            temperature=0.2,
        ),
        PhaseConfig(
            phase_type=PhaseType.ANNOTATE,
            model_type=OPENAI_04_MINI,
            temperature=0.1,
        ),
        PhaseConfig(
            phase_type=PhaseType.FINAL,
            model_type=OPENAI_04_MINI,
            temperature=0.1,
        ),
    ]

    config = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=run_phases,
        length_reduction=(30, 50),  # Conservative reduction for entire run
    )

    print(f"Run-level length reduction: {config.length_reduction}")
    print("This applies to all phases that use length reduction (edit and final)")
    # run_pipeline(config)  # Uncomment to run


def example_aggressive_reduction():
    """Example with aggressive length reduction."""
    print("Example 4: Aggressive reduction (60-80%)")

    run_phases = [
        PhaseConfig(
            phase_type=PhaseType.EDIT,
            model_type=OPENAI_04_MINI,
            temperature=0.2,
        ),
    ]

    config = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=run_phases,
        length_reduction=(60, 80),  # Aggressive reduction
    )

    print(f"Length reduction: {config.length_reduction}")
    # run_pipeline(config)  # Uncomment to run


def example_no_reduction():
    """Example with no length reduction specified."""
    print("Example 5: No length reduction specified")

    run_phases = [
        PhaseConfig(
            phase_type=PhaseType.EDIT,
            model_type=OPENAI_04_MINI,
            temperature=0.2,
        ),
    ]

    config = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=run_phases,
        # length_reduction=None (default)
    )

    print(f"Length reduction: {config.length_reduction}")
    print("Uses default values from prompt templates")
    # run_pipeline(config)  # Uncomment to run


def main():
    """Run all examples."""
    print("Length Reduction Parameter Examples (Run Level)")
    print("=" * 50)
    print()

    example_single_percentage()
    print()

    example_range()
    print()

    example_multiple_phases()
    print()

    example_aggressive_reduction()
    print()

    example_no_reduction()
    print()

    print("Note: All examples are configured but not executed.")
    print("Uncomment the run_pipeline(config) lines to actually run them.")
    print()
    print("Key points:")
    print("- length_reduction is set at the RunConfig level")
    print("- It applies to all phases that use length reduction (edit and final)")
    print("- Can be a single integer (40) or a tuple (30, 50)")
    print("- If not specified, uses default values from prompt templates")


if __name__ == "__main__":
    main()
