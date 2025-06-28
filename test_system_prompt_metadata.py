#!/usr/bin/env python3
"""
Test to verify system prompt metadata saving functionality.
"""

import json
import tempfile
from pathlib import Path

from src.config import PhaseConfig, PhaseType, RunConfig
from src.pipeline import Pipeline


def test_system_prompt_metadata():
    """Test that system prompt metadata is saved correctly."""

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        input_file = temp_path / "test_input.md"
        input_file.write_text("# Test Section\n\nThis is test content.")

        original_file = temp_path / "test_original.md"
        original_file.write_text("# Test Section\n\nThis is original content.")

        output_dir = temp_path / "output"
        output_dir.mkdir()

        # Create a simple configuration
        run_phases = [
            PhaseConfig(
                phase_type=PhaseType.EDIT,
                model_type="test-model",
                temperature=0.2,
            ),
        ]

        config = RunConfig(
            book_name="Test Book",
            author_name="Test Author",
            input_file=input_file,
            output_dir=output_dir,
            original_file=original_file,
            phases=run_phases,
            length_reduction=(30, 50),
        )

        # Create pipeline
        pipeline = Pipeline(config)

        # Initialize the first phase
        phase = pipeline._initialize_phase(0)

        if phase:
            # Save system prompt metadata
            pipeline._save_system_prompt_metadata(phase, 0)

            # Check that the metadata file was created
            metadata_files = list(output_dir.glob("system_prompt_metadata_*.json"))
            assert len(metadata_files) == 1, f"Expected 1 metadata file, found {len(metadata_files)}"

            # Read and verify the metadata
            with open(metadata_files[0], "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Verify the structure
            assert "run_timestamp" in metadata
            assert "book_name" in metadata
            assert "author_name" in metadata
            assert "length_reduction" in metadata
            assert "phase" in metadata

            phase_metadata = metadata["phase"]
            assert "phase_name" in phase_metadata
            assert "phase_index" in phase_metadata
            assert "phase_type" in phase_metadata
            assert "model_type" in phase_metadata
            assert "temperature" in phase_metadata
            assert "input_file" in phase_metadata
            assert "output_file" in phase_metadata
            assert "system_prompt_path" in phase_metadata
            assert "fully_rendered_system_prompt" in phase_metadata
            assert "length_reduction_parameter" in phase_metadata

            # Verify the values
            assert metadata["book_name"] == "Test Book"
            assert metadata["author_name"] == "Test Author"
            assert metadata["length_reduction"] == [30, 50]  # JSON serializes tuples as lists
            assert phase_metadata["phase_name"] == "edit"
            assert phase_metadata["phase_index"] == 0
            assert phase_metadata["phase_type"] == "EDIT"
            assert phase_metadata["model_type"] == "test-model"
            assert phase_metadata["temperature"] == 0.2
            assert phase_metadata["length_reduction_parameter"] == [30, 50]

            # Verify the system prompt contains the formatted length reduction
            system_prompt = phase_metadata["fully_rendered_system_prompt"]
            print(f"System prompt preview: {system_prompt[:500]}...")
            print(f"Length reduction parameter: {phase_metadata['length_reduction_parameter']}")
            print(f"Run length reduction: {metadata['length_reduction']}")

            # Check if the parameter is being formatted
            if "30-50%" in system_prompt:
                print("✓ Length reduction parameter correctly formatted in system prompt")
            elif "{length_reduction}" in system_prompt:
                print("✗ System prompt contains unformatted parameter placeholder")
            else:
                print("✗ System prompt does not contain length reduction information")

            assert "{length_reduction}" not in system_prompt, "System prompt should not contain unformatted parameter"

            print("✓ System prompt metadata test passed!")
            print(f"Metadata file: {metadata_files[0]}")
            print(f"System prompt length: {len(system_prompt)} characters")
        else:
            print("✗ Failed to initialize phase")


if __name__ == "__main__":
    test_system_prompt_metadata()
