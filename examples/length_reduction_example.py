#!/usr/bin/env python3
"""
Example demonstrating the length reduction parameter functionality.

This example shows how to use the length_reduction parameter to control
how much content is reduced during the edit and final phases.
"""

import sys
from pathlib import Path

# Add project root to path to allow importing from src
sys.path.append(str(Path(__file__).parent.parent))

from src.config import PhaseConfig, PhaseType, RunConfig


def main():
    """Demonstrate different length reduction configurations."""

    # Example 1: Single percentage reduction (40%)
    print("=== Example 1: Single percentage reduction (40%) ===")
    config1 = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=[PhaseConfig(phase_type=PhaseType.EDIT)],
        length_reduction=40,  # 40% reduction
    )

    # Example 2: Range of percentages (30-50% reduction)
    print("\n=== Example 2: Range reduction (30-50%) ===")
    config2 = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=[PhaseConfig(phase_type=PhaseType.EDIT)],
        length_reduction=(30, 50),  # 30-50% reduction
    )

    # Example 3: Use default (35-50% reduction)
    print("\n=== Example 3: Default reduction (35-50%) ===")
    config3 = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=[PhaseConfig(phase_type=PhaseType.EDIT)],
        # length_reduction defaults to (35, 50)
    )

    # Example 4: No reduction (set to None explicitly)
    print("\n=== Example 4: No reduction (None) ===")
    config4 = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("input.md"),
        output_dir=Path("output"),
        original_file=Path("original.md"),
        phases=[PhaseConfig(phase_type=PhaseType.EDIT)],
        length_reduction=None,  # No reduction specified
    )

    # Print configurations to show how they differ
    print(f"Config 1 length_reduction: {config1.length_reduction}")
    print(f"Config 2 length_reduction: {config2.length_reduction}")
    print(f"Config 3 length_reduction: {config3.length_reduction}")
    print(f"Config 4 length_reduction: {config4.length_reduction}")

    # Note: In a real scenario, you would run the pipeline with one of these configs
    # pipeline = Pipeline(config1)
    # result = pipeline.run()


if __name__ == "__main__":
    main()
