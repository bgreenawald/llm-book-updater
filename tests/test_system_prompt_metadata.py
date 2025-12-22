#!/usr/bin/env python3
"""
Test to verify consolidated pipeline metadata saving functionality.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path to allow importing from src
sys.path.append(str(Path(__file__).parent.parent))

from src.api.config import PhaseConfig, PhaseType, RunConfig
from src.core.pipeline import Pipeline


@patch("src.models.model.LlmModel.create")
@patch("src.models.cost_tracking.CostTrackingWrapper")
def test_pipeline_metadata(mock_cost_wrapper, mock_llm_create):
    """Test that pipeline metadata is correctly collected and saved."""
    # Mock the LlmModel.create to avoid API key requirements
    mock_model_instance = Mock()
    mock_model_instance.model_id = "test/model"
    mock_llm_create.return_value = mock_model_instance

    # Mock the cost wrapper to avoid API dependencies
    mock_wrapper_instance = Mock()
    mock_wrapper_instance.enabled = False
    mock_cost_wrapper.return_value = mock_wrapper_instance

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
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=original_file,
                phases=[
                    PhaseConfig(
                        phase_type=PhaseType.MODERNIZE,
                        enabled=True,
                    )
                ],
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
            assert "phases" in metadata

            # Verify metadata version
            assert metadata["metadata_version"] == "0.0"

            # Verify phase metadata
            assert len(metadata["phases"]) == 1
            phase_metadata = metadata["phases"][0]

            assert "phase_name" in phase_metadata
            assert "phase_index" in phase_metadata
            assert "phase_type" in phase_metadata
            assert "enabled" in phase_metadata
            assert "model_type" in phase_metadata
            assert "max_workers" in phase_metadata
            assert "input_file" in phase_metadata
            assert "output_file" in phase_metadata
            assert "system_prompt_path" in phase_metadata
            assert "user_prompt_path" in phase_metadata
            assert "fully_rendered_system_prompt" in phase_metadata
            assert "post_processors" in phase_metadata
            assert "post_processor_count" in phase_metadata
            assert "completed" in phase_metadata
            assert "output_exists" in phase_metadata

            # Verify specific values
            assert phase_metadata["phase_index"] == 0
            assert phase_metadata["book_name"] == "Test Book"
            assert phase_metadata["author_name"] == "Test Author"
            assert phase_metadata["completed"] is True

            # Verify system prompt content
            system_prompt = phase_metadata["fully_rendered_system_prompt"]
            assert isinstance(system_prompt, str)
            assert len(system_prompt) > 0

            print("✓ Pipeline metadata test passed")

    except Exception as e:
        print(f"✗ Pipeline metadata test failed: {e}")
        raise


@patch("src.models.model.LlmModel.create")
@patch("src.models.cost_tracking.CostTrackingWrapper")
def test_cost_analysis_saving(mock_cost_wrapper, mock_llm_create):
    """Test that cost analysis data is correctly saved."""
    # Mock the LlmModel.create to avoid API key requirements
    mock_model_instance = Mock()
    mock_model_instance.model_id = "test/model"
    mock_llm_create.return_value = mock_model_instance

    # Mock the cost wrapper to avoid API dependencies
    mock_wrapper_instance = Mock()
    mock_wrapper_instance.enabled = False
    mock_cost_wrapper.return_value = mock_wrapper_instance

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
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=original_file,
                phases=[
                    PhaseConfig(
                        phase_type=PhaseType.MODERNIZE,
                        enabled=True,
                    )
                ],
            )

            # Create pipeline
            pipeline = Pipeline(config=test_config)

            # Test cost analysis saving with mock data
            mock_cost_analysis = {
                "total_phases": 1,
                "completed_phases": 1,
                "total_generations": 5,
                "total_prompt_tokens": 1000,
                "total_completion_tokens": 500,
                "total_tokens": 1500,
                "total_cost": 0.0025,
                "currency": "USD",
                "phase_costs": [
                    {
                        "phase_name": "modernize",
                        "phase_index": 0,
                        "generation_ids": ["gen_1", "gen_2"],
                        "total_prompt_tokens": 1000,
                        "total_completion_tokens": 500,
                        "total_tokens": 1500,
                        "total_cost": 0.0025,
                        "currency": "USD",
                        "generation_count": 2,
                    }
                ],
            }

            # Save cost analysis
            pipeline._save_cost_analysis(cost_analysis=mock_cost_analysis)

            # Check that cost analysis file was created
            cost_files = list(output_dir.glob(pattern="cost_analysis_*.json"))
            assert len(cost_files) == 1, f"Expected 1 cost analysis file, found {len(cost_files)}"

            # Read and verify cost analysis
            with open(file=cost_files[0], mode="r", encoding="utf-8") as f:
                cost_data = json.load(fp=f)

            # Verify cost analysis structure
            assert "total_phases" in cost_data
            assert "completed_phases" in cost_data
            assert "total_generations" in cost_data
            assert "total_prompt_tokens" in cost_data
            assert "total_completion_tokens" in cost_data
            assert "total_tokens" in cost_data
            assert "total_cost" in cost_data
            assert "currency" in cost_data
            assert "phase_costs" in cost_data

            # Verify specific values
            assert cost_data["total_phases"] == 1
            assert cost_data["completed_phases"] == 1
            assert cost_data["total_generations"] == 5
            assert cost_data["total_prompt_tokens"] == 1000
            assert cost_data["total_completion_tokens"] == 500
            assert cost_data["total_tokens"] == 1500
            assert cost_data["total_cost"] == 0.0025
            assert cost_data["currency"] == "USD"

            # Verify phase costs
            assert len(cost_data["phase_costs"]) == 1
            phase_cost = cost_data["phase_costs"][0]
            assert phase_cost["phase_name"] == "modernize"
            assert phase_cost["phase_index"] == 0
            assert phase_cost["generation_ids"] == ["gen_1", "gen_2"]
            assert phase_cost["generation_count"] == 2

            print("✓ Cost analysis saving test passed")

    except Exception as e:
        print(f"✗ Cost analysis saving test failed: {e}")
        raise


@patch("src.models.model.LlmModel.create")
@patch("src.models.cost_tracking.CostTrackingWrapper")
def test_metadata_with_disabled_phases(mock_cost_wrapper, mock_llm_create):
    """Test that metadata is correctly collected for disabled phases."""
    # Mock the LlmModel.create to avoid API key requirements
    mock_model_instance = Mock()
    mock_model_instance.model_id = "test/model"
    mock_llm_create.return_value = mock_model_instance

    # Mock the cost wrapper to avoid API dependencies
    mock_wrapper_instance = Mock()
    mock_wrapper_instance.enabled = False
    mock_cost_wrapper.return_value = mock_wrapper_instance

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

            # Create a test config with enabled and disabled phases
            test_config = RunConfig(
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=original_file,
                phases=[
                    PhaseConfig(
                        phase_type=PhaseType.MODERNIZE,
                        enabled=True,
                    ),
                    PhaseConfig(
                        phase_type=PhaseType.EDIT,
                        enabled=False,  # Disabled phase
                    ),
                    PhaseConfig(
                        phase_type=PhaseType.ANNOTATE,
                        enabled=True,
                    ),
                ],
            )

            # Create pipeline
            pipeline = Pipeline(config=test_config)

            # Collect metadata for disabled phase
            pipeline._collect_phase_metadata(phase=None, phase_index=1, completed=False)

            # Verify that disabled phase metadata was collected
            assert len(pipeline._phase_metadata) == 1
            disabled_phase_metadata = pipeline._phase_metadata[0]

            # Verify disabled phase metadata structure
            assert disabled_phase_metadata["phase_index"] == 1
            assert disabled_phase_metadata["phase_type"] == "EDIT"
            assert disabled_phase_metadata["enabled"] is False
            assert disabled_phase_metadata["completed"] is False
            assert disabled_phase_metadata["reason"] == "disabled"

            print("✓ Disabled phase metadata test passed")

    except Exception as e:
        print(f"✗ Disabled phase metadata test failed: {e}")
        raise


@patch("src.models.model.LlmModel.create")
@patch("src.models.cost_tracking.CostTrackingWrapper")
def test_metadata_with_failed_phases(mock_cost_wrapper, mock_llm_create):
    """Test that metadata is correctly collected for failed phases."""
    # Mock the LlmModel.create to avoid API key requirements
    mock_model_instance = Mock()
    mock_model_instance.model_id = "test/model"
    mock_llm_create.return_value = mock_model_instance

    # Mock the cost wrapper to avoid API dependencies
    mock_wrapper_instance = Mock()
    mock_wrapper_instance.enabled = False
    mock_cost_wrapper.return_value = mock_wrapper_instance

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
            test_config = RunConfig(
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=original_file,
                phases=[
                    PhaseConfig(
                        phase_type=PhaseType.MODERNIZE,
                        enabled=True,
                    )
                ],
            )

            # Create pipeline
            pipeline = Pipeline(config=test_config)

            # Initialize phase
            phase = pipeline._initialize_phase(phase_index=0)
            assert phase is not None

            # Collect metadata for failed phase
            pipeline._collect_phase_metadata(phase=phase, phase_index=0, completed=False)

            # Verify that failed phase metadata was collected
            assert len(pipeline._phase_metadata) == 1
            failed_phase_metadata = pipeline._phase_metadata[0]

            # Verify failed phase metadata structure
            assert failed_phase_metadata["phase_index"] == 0
            assert failed_phase_metadata["phase_type"] == "MODERNIZE"
            assert failed_phase_metadata["enabled"] is True
            assert failed_phase_metadata["completed"] is False
            assert failed_phase_metadata["reason"] == "not_run"

            print("✓ Failed phase metadata test passed")

    except Exception as e:
        print(f"✗ Failed phase metadata test failed: {e}")
        raise


if __name__ == "__main__":
    test_pipeline_metadata()
    test_cost_analysis_saving()
    test_metadata_with_disabled_phases()
    test_metadata_with_failed_phases()
