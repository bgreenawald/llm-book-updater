"""
Tests for metadata consolidation functionality.

These tests verify that multiple metadata files can be consolidated correctly.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.processing.metadata_consolidator import MetadataConsolidator, consolidate_metadata


class TestMetadataConsolidator:
    """Tests for the MetadataConsolidator class."""

    def test_consolidator_init_valid_directory(self):
        """Test that consolidator initializes with a valid directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            consolidator = MetadataConsolidator(temp_path)
            assert consolidator.output_dir == temp_path

    def test_consolidator_init_invalid_directory(self):
        """Test that consolidator raises error for non-existent directory."""
        with pytest.raises(ValueError) as exc_info:
            MetadataConsolidator(Path("/nonexistent/directory"))
        assert "does not exist" in str(exc_info.value)

    def test_find_metadata_files_empty_directory(self):
        """Test finding metadata files in empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            consolidator = MetadataConsolidator(temp_path)
            files = consolidator.find_metadata_files()
            assert files == []

    def test_find_metadata_files_with_metadata(self):
        """Test finding metadata files when they exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create dummy metadata files
            (temp_path / "pipeline_metadata_20250101_120000.json").write_text("{}")
            (temp_path / "pipeline_metadata_20250102_120000.json").write_text("{}")
            (temp_path / "other_file.json").write_text("{}")

            consolidator = MetadataConsolidator(temp_path)
            files = consolidator.find_metadata_files()

            assert len(files) == 2
            assert all("pipeline_metadata_" in f.name for f in files)

    def test_load_metadata_file_valid(self):
        """Test loading a valid metadata file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            metadata_file = temp_path / "test_metadata.json"

            test_data = {"key": "value", "number": 42}
            metadata_file.write_text(json.dumps(test_data))

            consolidator = MetadataConsolidator(temp_path)
            loaded = consolidator.load_metadata_file(metadata_file)

            assert loaded == test_data

    def test_load_metadata_file_invalid_json(self):
        """Test loading an invalid JSON file returns None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            metadata_file = temp_path / "invalid.json"
            metadata_file.write_text("not valid json{")

            consolidator = MetadataConsolidator(temp_path)
            loaded = consolidator.load_metadata_file(metadata_file)

            assert loaded is None

    def test_consolidate_no_files_raises_error(self):
        """Test that consolidating with no files raises an error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            consolidator = MetadataConsolidator(temp_path)

            with pytest.raises(ValueError) as exc_info:
                consolidator.consolidate_metadata([])
            assert "No metadata files provided" in str(exc_info.value)

    def test_consolidate_single_file(self):
        """Test consolidating a single metadata file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            metadata = {
                "metadata_version": "1.0.0",
                "run_timestamp": "2025-01-01T12:00:00",
                "book_id": "test_book",
                "phases": [
                    {"phase_index": 0, "phase_type": "MODERNIZE", "completed": True},
                    {"phase_index": 1, "phase_type": "EDIT", "completed": True},
                ],
            }

            metadata_file = temp_path / "pipeline_metadata_20250101.json"
            metadata_file.write_text(json.dumps(metadata))

            consolidator = MetadataConsolidator(temp_path)
            result = consolidator.consolidate_metadata([metadata_file])

            assert result["book_id"] == "test_book"
            assert len(result["phases"]) == 2
            assert "consolidation_info" in result

    def test_consolidate_multiple_files_all_completed(self):
        """Test consolidating multiple files where different phases completed in each run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # First run: phases 0-2 completed
            metadata1 = {
                "metadata_version": "1.0.0",
                "run_timestamp": "2025-01-01T12:00:00",
                "book_id": "test_book",
                "phases": [
                    {"phase_index": 0, "phase_type": "MODERNIZE", "completed": True},
                    {"phase_index": 1, "phase_type": "EDIT", "completed": True},
                    {"phase_index": 2, "phase_type": "FINAL", "completed": True},
                    {"phase_index": 3, "phase_type": "INTRODUCTION", "completed": False, "reason": "not_run"},
                ],
            }

            # Second run: phases 0-2 skipped, phase 3 completed
            metadata2 = {
                "metadata_version": "1.0.0",
                "run_timestamp": "2025-01-02T12:00:00",
                "book_id": "test_book",
                "phases": [
                    {"phase_index": 0, "phase_type": "MODERNIZE", "completed": False, "reason": "skipped"},
                    {"phase_index": 1, "phase_type": "EDIT", "completed": False, "reason": "skipped"},
                    {"phase_index": 2, "phase_type": "FINAL", "completed": False, "reason": "skipped"},
                    {"phase_index": 3, "phase_type": "INTRODUCTION", "completed": True},
                ],
            }

            file1 = temp_path / "pipeline_metadata_20250101.json"
            file2 = temp_path / "pipeline_metadata_20250102.json"
            file1.write_text(json.dumps(metadata1))
            file2.write_text(json.dumps(metadata2))

            consolidator = MetadataConsolidator(temp_path)
            result = consolidator.consolidate_metadata([file1, file2])

            # All phases should be marked as completed
            assert len(result["phases"]) == 4
            assert all(phase["completed"] for phase in result["phases"])

    def test_consolidate_prefers_completed_over_skipped(self):
        """Test that consolidation prefers completed phase data over skipped."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # First run: phase 0 completed
            metadata1 = {
                "book_id": "test_book",
                "phases": [
                    {
                        "phase_index": 0,
                        "phase_type": "MODERNIZE",
                        "completed": True,
                        "input_file": "input.md",
                        "output_file": "output.md",
                    }
                ],
            }

            # Second run: phase 0 skipped
            metadata2 = {
                "book_id": "test_book",
                "phases": [
                    {
                        "phase_index": 0,
                        "phase_type": "MODERNIZE",
                        "completed": False,
                        "reason": "skipped",
                        "input_file": None,
                        "output_file": None,
                    }
                ],
            }

            file1 = temp_path / "pipeline_metadata_20250101.json"
            file2 = temp_path / "pipeline_metadata_20250102.json"
            file1.write_text(json.dumps(metadata1))
            file2.write_text(json.dumps(metadata2))

            consolidator = MetadataConsolidator(temp_path)
            result = consolidator.consolidate_metadata([file1, file2])

            # Should use the completed phase data from run 1
            assert result["phases"][0]["completed"] is True
            assert result["phases"][0]["input_file"] == "input.md"
            assert result["phases"][0]["output_file"] == "output.md"

    def test_consolidate_handles_different_phase_counts(self):
        """Test consolidation with different numbers of phases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # First run: 2 phases
            metadata1 = {
                "book_id": "test_book",
                "phases": [
                    {"phase_index": 0, "phase_type": "MODERNIZE", "completed": True},
                    {"phase_index": 1, "phase_type": "EDIT", "completed": True},
                ],
            }

            # Second run: 4 phases
            metadata2 = {
                "book_id": "test_book",
                "phases": [
                    {"phase_index": 0, "phase_type": "MODERNIZE", "completed": False, "reason": "skipped"},
                    {"phase_index": 1, "phase_type": "EDIT", "completed": False, "reason": "skipped"},
                    {"phase_index": 2, "phase_type": "FINAL", "completed": True},
                    {"phase_index": 3, "phase_type": "INTRODUCTION", "completed": True},
                ],
            }

            file1 = temp_path / "pipeline_metadata_20250101.json"
            file2 = temp_path / "pipeline_metadata_20250102.json"
            file1.write_text(json.dumps(metadata1))
            file2.write_text(json.dumps(metadata2))

            consolidator = MetadataConsolidator(temp_path)
            result = consolidator.consolidate_metadata([file1, file2])

            # Should have all 4 phases, all completed
            assert len(result["phases"]) == 4
            assert all(phase["completed"] for phase in result["phases"])

    def test_save_consolidated_metadata(self):
        """Test saving consolidated metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            consolidator = MetadataConsolidator(temp_path)
            test_data = {"test": "data", "phases": []}

            output_path = consolidator.save_consolidated_metadata(test_data, "test_output.json")

            assert output_path.exists()
            with open(output_path, "r") as f:
                loaded = json.load(f)
            assert loaded == test_data

    def test_run_end_to_end(self):
        """Test the complete consolidation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple metadata files
            metadata1 = {
                "book_id": "test_book",
                "phases": [
                    {"phase_index": 0, "completed": True},
                    {"phase_index": 1, "completed": True},
                ],
            }

            metadata2 = {
                "book_id": "test_book",
                "phases": [
                    {"phase_index": 0, "completed": False, "reason": "skipped"},
                    {"phase_index": 1, "completed": False, "reason": "skipped"},
                    {"phase_index": 2, "completed": True},
                ],
            }

            (temp_path / "pipeline_metadata_001.json").write_text(json.dumps(metadata1))
            (temp_path / "pipeline_metadata_002.json").write_text(json.dumps(metadata2))

            consolidator = MetadataConsolidator(temp_path)
            output_path = consolidator.run()

            assert output_path.exists()
            with open(output_path, "r") as f:
                result = json.load(f)

            assert len(result["phases"]) == 3
            assert all(phase["completed"] for phase in result["phases"])
            assert "consolidation_info" in result

    def test_convenience_function(self):
        """Test the convenience function for consolidation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            metadata = {
                "book_id": "test_book",
                "phases": [{"phase_index": 0, "completed": True}],
            }

            (temp_path / "pipeline_metadata_001.json").write_text(json.dumps(metadata))

            output_path = consolidate_metadata(temp_path)

            assert output_path.exists()
            assert output_path.name == "pipeline_metadata_consolidated.json"
