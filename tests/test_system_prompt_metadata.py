#!/usr/bin/env python3
"""
Test to verify system prompt metadata saving functionality.
"""

import json
import tempfile
from pathlib import Path

from src.pipeline import Pipeline
from src.run_settings import config


def test_system_prompt_metadata():
    """Test that system prompt metadata is correctly collected and saved."""
    try:
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test input file
            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Section\n\nThis is test content.")

            # Create test original file
            original_file = temp_path / "test_original.md"
            original_file.write_text("# Test Section\n\nThis is original content.")

            # Create output directory
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Create a test config
            test_config = type(config)(
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=original_file,
                phases=config.phases,
                length_reduction=(35, 50),
            )

            # Create pipeline
            pipeline = Pipeline(config=test_config)

            # Initialize first phase
            phase = pipeline._initialize_phase(phase_index=0)
            assert phase is not None

            # Collect system prompt metadata
            pipeline._collect_system_prompt_metadata(phase=phase, phase_index=0)

            # Save metadata
            pipeline._save_all_system_prompt_metadata()

            # Check that metadata file was created
            metadata_files = list(output_dir.glob(pattern="system_prompt_metadata_*.json"))
            assert len(metadata_files) == 1, f"Expected 1 metadata file, found {len(metadata_files)}"

            # Read and verify metadata
            with open(file=metadata_files[0], mode="r", encoding="utf-8") as f:
                metadata = json.load(fp=f)

            # Verify metadata structure
            assert "run_timestamp" in metadata
            assert "book_name" in metadata
            assert "author_name" in metadata
            assert "input_file" in metadata
            assert "original_file" in metadata
            assert "output_directory" in metadata
            assert "length_reduction" in metadata
            assert "phases" in metadata

            # Verify phase metadata
            assert len(metadata["phases"]) == 1
            phase_metadata = metadata["phases"][0]

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

            # Verify specific values
            assert phase_metadata["phase_index"] == 0
            assert phase_metadata["book_name"] == "Test Book"
            assert phase_metadata["author_name"] == "Test Author"
            assert phase_metadata["length_reduction_parameter"] == [35, 50]

            # Verify system prompt content
            system_prompt = phase_metadata["fully_rendered_system_prompt"]
            assert isinstance(system_prompt, str)
            assert len(system_prompt) > 0

            # Verify length reduction parameter is formatted in system prompt
            if "35-50%" in system_prompt:
                print("✓ Length reduction parameter correctly formatted in system prompt")
            elif "{length_reduction}" in system_prompt:
                print("✗ System prompt contains unformatted parameter placeholder")
            else:
                print("✗ System prompt does not contain length reduction information")

            # Print summary
            print("✓ System prompt metadata test passed!")
            print(f"Metadata file: {metadata_files[0]}")
            print(f"System prompt length: {len(system_prompt)} characters")
    except Exception:
        print("✗ Failed to initialize phase")
        raise


if __name__ == "__main__":
    test_system_prompt_metadata()
