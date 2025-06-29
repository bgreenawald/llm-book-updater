#!/usr/bin/env python3
"""
Example demonstrating the metadata saving functionality of the pipeline.

This example shows how metadata is automatically saved when running the pipeline.
"""

from pathlib import Path

from src.config import PhaseConfig, PhaseType, RunConfig
from src.pipeline import Pipeline


def main():
    """Demonstrate metadata saving functionality."""

    # Create a simple configuration
    config = RunConfig(
        book_name="Example Book",
        author_name="Example Author",
        input_file=Path("examples/sample_input.txt"),
        output_dir=Path("examples/output"),
        original_file=Path("examples/original.txt"),
        phases=[
            PhaseConfig(phase_type=PhaseType.MODERNIZE, enabled=True, temperature=0.2),
            PhaseConfig(
                phase_type=PhaseType.EDIT,
                enabled=False,  # This phase will be skipped
                temperature=0.3,
            ),
            PhaseConfig(phase_type=PhaseType.ANNOTATE, enabled=True, temperature=0.1),
        ],
    )

    # Create and run the pipeline
    pipeline = Pipeline(config)

    print("Running pipeline with metadata saving...")
    print(f"Output directory: {config.output_dir}")

    # Note: This will fail if the input file doesn't exist, but metadata will still be saved
    try:
        pipeline.run()
        print("✓ Pipeline completed - metadata saved once for the entire run")
    except FileNotFoundError:
        print("Input file not found, but metadata saving functionality is demonstrated.")
        print("Check the output directory for metadata files.")

    print("\nMetadata files will be saved as:")
    print("run_metadata_YYYYMMDD_HHMMSS.json")
    print("\nThe metadata includes:")
    print("- Run timestamp")
    print("- Book and author information")
    print("- Input and output file paths")
    print("- Phase configurations and completion status")
    print("- Model settings for each phase")

    print("\nNote: Metadata is saved ONCE per complete pipeline run, not for each individual phase.")
    print("To save metadata for individual phases, use: pipeline.run_phase(phase_type, save_metadata=True)")


if __name__ == "__main__":
    main()
