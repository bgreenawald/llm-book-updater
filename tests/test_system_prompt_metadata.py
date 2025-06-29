#!/usr/bin/env python3
"""
Test to verify consolidated pipeline metadata saving functionality.
"""

import json
import tempfile
from pathlib import Path

from src.config import PhaseConfig, PhaseType, RunConfig
from src.pipeline import Pipeline


def test_pipeline_metadata():
    """Test that pipeline metadata is correctly collected and saved."""
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

            # Create a test config with a simple phase
            test_config = RunConfig(
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=original_file,
                phases=[
                    PhaseConfig(
                        phase_type=PhaseType.MODERNIZE,
                        enabled=True,
                        temperature=0.2,
                    )
                ],
                length_reduction=(35, 50),
            )

            # Create pipeline
            pipeline = Pipeline(config=test_config)

            # Initialize first phase
            phase = pipeline._initialize_phase(phase_index=0)
            assert phase is not None

            # Collect phase metadata
            pipeline._collect_phase_metadata(phase=phase, phase_index=0, completed=True)

            # Save metadata
            pipeline._save_metadata(completed_phases=[phase])

            # Check that metadata file was created
            metadata_files = list(output_dir.glob(pattern="pipeline_metadata_*.json"))
            assert len(metadata_files) == 1, f"Expected 1 metadata file, found {len(metadata_files)}"

            # Read and verify metadata
            with open(file=metadata_files[0], mode="r", encoding="utf-8") as f:
                metadata = json.load(fp=f)

            # Verify metadata structure
            assert "metadata_version" in metadata
            assert "run_timestamp" in metadata
            assert "book_name" in metadata
            assert "author_name" in metadata
            assert "input_file" in metadata
            assert "original_file" in metadata
            assert "output_directory" in metadata
            assert "length_reduction" in metadata
            assert "phases" in metadata

            # Verify metadata version
            assert metadata["metadata_version"] == "1.0.0"

            # Verify phase metadata
            assert len(metadata["phases"]) == 1
            phase_metadata = metadata["phases"][0]

            assert "phase_name" in phase_metadata
            assert "phase_index" in phase_metadata
            assert "phase_type" in phase_metadata
            assert "enabled" in phase_metadata
            assert "model_type" in phase_metadata
            assert "temperature" in phase_metadata
            assert "max_workers" in phase_metadata
            assert "input_file" in phase_metadata
            assert "output_file" in phase_metadata
            assert "system_prompt_path" in phase_metadata
            assert "user_prompt_path" in phase_metadata
            assert "fully_rendered_system_prompt" in phase_metadata
            assert "length_reduction_parameter" in phase_metadata
            assert "post_processors" in phase_metadata
            assert "post_processor_count" in phase_metadata
            assert "completed" in phase_metadata
            assert "output_exists" in phase_metadata

            # Verify specific values
            assert phase_metadata["phase_index"] == 0
            assert phase_metadata["book_name"] == "Test Book"
            assert phase_metadata["author_name"] == "Test Author"
            assert phase_metadata["length_reduction_parameter"] == [35, 50]
            assert phase_metadata["completed"] is True

            # Verify system prompt content
            system_prompt = phase_metadata["fully_rendered_system_prompt"]
            assert isinstance(system_prompt, str)
            assert len(system_prompt) > 0

            print("✓ Pipeline metadata test passed")

    except Exception as e:
        print(f"✗ Pipeline metadata test failed: {e}")
        raise


if __name__ == "__main__":
    test_pipeline_metadata()
