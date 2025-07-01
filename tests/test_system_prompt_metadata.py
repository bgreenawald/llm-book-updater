#!/usr/bin/env python3
"""
Test to verify consolidated pipeline metadata saving functionality.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path to allow importing from src
sys.path.append(str(Path(__file__).parent.parent))

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
            assert metadata["metadata_version"] == "0.0.0-alpha"

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


def test_cost_analysis_saving():
    """Test that cost analysis is correctly saved as a separate JSON file."""
    try:
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create output directory
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Create a test config
            test_config = RunConfig(
                book_name="Test Book",
                author_name="Test Author",
                input_file=temp_path / "test_input.md",
                output_dir=output_dir,
                original_file=temp_path / "test_original.md",
                phases=[],
            )

            # Create pipeline
            pipeline = Pipeline(config=test_config)

            # Create sample cost analysis data
            cost_analysis = {
                "total_phases": 2,
                "completed_phases": 2,
                "total_generations": 5,
                "total_prompt_tokens": 1000,
                "total_completion_tokens": 2000,
                "total_tokens": 3000,
                "total_cost": 0.015,
                "currency": "USD",
                "phase_costs": [
                    {
                        "phase_name": "modernize",
                        "phase_index": 0,
                        "generation_ids": ["gen1", "gen2"],
                        "total_prompt_tokens": 500,
                        "total_completion_tokens": 1000,
                        "total_tokens": 1500,
                        "total_cost": 0.0075,
                        "currency": "USD",
                        "generation_count": 2,
                    },
                    {
                        "phase_name": "edit",
                        "phase_index": 1,
                        "generation_ids": ["gen3", "gen4", "gen5"],
                        "total_prompt_tokens": 500,
                        "total_completion_tokens": 1000,
                        "total_tokens": 1500,
                        "total_cost": 0.0075,
                        "currency": "USD",
                        "generation_count": 3,
                    },
                ],
            }

            # Save cost analysis
            pipeline._save_cost_analysis(cost_analysis=cost_analysis)

            # Check that cost analysis file was created
            cost_files = list(output_dir.glob(pattern="cost_analysis_*.json"))
            assert len(cost_files) == 1, f"Expected 1 cost analysis file, found {len(cost_files)}"

            # Read and verify cost analysis
            with open(file=cost_files[0], mode="r", encoding="utf-8") as f:
                saved_cost_analysis = json.load(fp=f)

            # Verify cost analysis structure
            assert "total_phases" in saved_cost_analysis
            assert "completed_phases" in saved_cost_analysis
            assert "total_generations" in saved_cost_analysis
            assert "total_prompt_tokens" in saved_cost_analysis
            assert "total_completion_tokens" in saved_cost_analysis
            assert "total_tokens" in saved_cost_analysis
            assert "total_cost" in saved_cost_analysis
            assert "currency" in saved_cost_analysis
            assert "phase_costs" in saved_cost_analysis

            # Verify specific values
            assert saved_cost_analysis["total_phases"] == 2
            assert saved_cost_analysis["completed_phases"] == 2
            assert saved_cost_analysis["total_generations"] == 5
            assert saved_cost_analysis["total_prompt_tokens"] == 1000
            assert saved_cost_analysis["total_completion_tokens"] == 2000
            assert saved_cost_analysis["total_tokens"] == 3000
            assert saved_cost_analysis["total_cost"] == 0.015
            assert saved_cost_analysis["currency"] == "USD"

            # Verify phase costs
            assert len(saved_cost_analysis["phase_costs"]) == 2
            phase1 = saved_cost_analysis["phase_costs"][0]
            assert phase1["phase_name"] == "modernize"
            assert phase1["phase_index"] == 0
            assert phase1["generation_count"] == 2
            assert phase1["total_cost"] == 0.0075

            phase2 = saved_cost_analysis["phase_costs"][1]
            assert phase2["phase_name"] == "edit"
            assert phase2["phase_index"] == 1
            assert phase2["generation_count"] == 3
            assert phase2["total_cost"] == 0.0075

            print("✓ Cost analysis saving test passed")

    except Exception as e:
        print(f"✗ Cost analysis saving test failed: {e}")
        raise


if __name__ == "__main__":
    test_pipeline_metadata()
    test_cost_analysis_saving()
