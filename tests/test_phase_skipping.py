"""
Tests for phase skipping functionality.

These tests verify that the pipeline can skip phases and resume from a specific phase
when configured with start_from_phase parameter.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.config import PhaseConfig, PhaseType, RunConfig
from src.pipeline import Pipeline


class TestPhaseSkipping:
    """Tests for phase skipping and resumption functionality."""

    def test_start_from_phase_zero_is_default(self):
        """Test that start_from_phase defaults to 0."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            config = RunConfig(
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=input_file,
                phases=[
                    PhaseConfig(phase_type=PhaseType.MODERNIZE),
                    PhaseConfig(phase_type=PhaseType.EDIT),
                ],
            )

            assert config.start_from_phase == 0

    def test_start_from_phase_can_be_set(self):
        """Test that start_from_phase can be set to a custom value."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            config = RunConfig(
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=input_file,
                phases=[
                    PhaseConfig(phase_type=PhaseType.MODERNIZE),
                    PhaseConfig(phase_type=PhaseType.EDIT),
                    PhaseConfig(phase_type=PhaseType.FINAL),
                ],
                start_from_phase=2,
            )

            assert config.start_from_phase == 2

    def test_start_from_phase_negative_raises_error(self):
        """Test that negative start_from_phase raises an error during config creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            with pytest.raises(ValidationError) as exc_info:
                RunConfig(
                    book_id="test_book",
                    book_name="Test Book",
                    author_name="Test Author",
                    input_file=input_file,
                    output_dir=output_dir,
                    original_file=input_file,
                    phases=[
                        PhaseConfig(phase_type=PhaseType.MODERNIZE),
                        PhaseConfig(phase_type=PhaseType.EDIT),
                    ],
                    start_from_phase=-1,
                )

            assert "start_from_phase" in str(exc_info.value)

    def test_start_from_phase_out_of_bounds_raises_error(self):
        """Test that start_from_phase > len(phases) raises an error during config creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            with pytest.raises(ValidationError) as exc_info:
                RunConfig(
                    book_id="test_book",
                    book_name="Test Book",
                    author_name="Test Author",
                    input_file=input_file,
                    output_dir=output_dir,
                    original_file=input_file,
                    phases=[
                        PhaseConfig(phase_type=PhaseType.MODERNIZE),
                        PhaseConfig(phase_type=PhaseType.EDIT),
                    ],
                    start_from_phase=5,  # Out of bounds (only 2 phases)
                )

            assert "out of range" in str(exc_info.value) or "start_from_phase" in str(exc_info.value)

    def test_start_from_phase_without_previous_output_raises_error(self):
        """Test that starting from a phase without previous phase output raises an error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            config = RunConfig(
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=input_file,
                phases=[
                    PhaseConfig(phase_type=PhaseType.MODERNIZE),
                    PhaseConfig(phase_type=PhaseType.EDIT),
                    PhaseConfig(phase_type=PhaseType.FINAL),
                ],
                start_from_phase=2,  # Starting from phase 2 but no phase 1 output
            )

            pipeline = Pipeline(config=config)
            with pytest.raises(ValueError) as exc_info:
                pipeline.run()

            assert "required input file not found" in str(exc_info.value)
            assert "previous phase" in str(exc_info.value)

    @patch("src.pipeline.Pipeline._initialize_phase")
    def test_start_from_phase_skips_earlier_phases(self, mock_initialize_phase):
        """Test that phases before start_from_phase are skipped."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content\n\n## Section 1\n\nSome text here.")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Create fake output from phase 0 and 1
            phase_0_output = output_dir / "01-test_input Modernize_1.md"
            phase_0_output.write_text("# Test Content\n\n## Section 1\n\nSome modernized text.")

            phase_1_output = output_dir / "02-test_input Edit_1.md"
            phase_1_output.write_text("# Test Content\n\n## Section 1\n\nSome edited text.")

            config = RunConfig(
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=input_file,
                phases=[
                    PhaseConfig(phase_type=PhaseType.MODERNIZE),
                    PhaseConfig(phase_type=PhaseType.EDIT),
                    PhaseConfig(phase_type=PhaseType.FINAL),
                ],
                start_from_phase=2,  # Start from FINAL phase
            )

            # Mock the phase initialization to avoid LLM calls
            mock_phase = MagicMock()
            mock_phase.name = "final"
            mock_phase.output_file_path = output_dir / "03-test_input Final_1.md"
            mock_initialize_phase.return_value = mock_phase

            pipeline = Pipeline(config=config)
            pipeline.run()

            # Verify that only phase 2 was initialized (phases 0 and 1 were skipped)
            assert mock_initialize_phase.call_count == 1
            # Verify the call was for phase 2
            mock_initialize_phase.assert_called_with(phase_index=2)

    @patch("src.pipeline.Pipeline._initialize_phase")
    def test_skipped_phases_metadata_has_skipped_reason(self, mock_initialize_phase):
        """Test that skipped phases have 'skipped' in their metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Create fake output from phase 0
            phase_0_output = output_dir / "01-test_input Modernize_1.md"
            phase_0_output.write_text("# Test Content (modernized)")

            config = RunConfig(
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=input_file,
                phases=[
                    PhaseConfig(phase_type=PhaseType.MODERNIZE),
                    PhaseConfig(phase_type=PhaseType.EDIT),
                ],
                start_from_phase=1,
            )

            # Mock the phase to avoid LLM calls
            mock_phase = MagicMock()
            mock_phase.name = "edit"
            mock_phase.output_file_path = output_dir / "02-test_input Edit_1.md"
            mock_initialize_phase.return_value = mock_phase

            pipeline = Pipeline(config=config)
            pipeline.run()

            # Check metadata for skipped phases
            assert len(pipeline._phase_metadata) == 2
            # Phase 0 should be marked as skipped
            assert pipeline._phase_metadata[0]["reason"] == "skipped"
            assert pipeline._phase_metadata[0]["completed"] is False

    def test_start_from_phase_zero_copies_input_file(self):
        """Test that starting from phase 0 copies the input file to output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            config = RunConfig(
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=input_file,
                phases=[],
                start_from_phase=0,
            )

            pipeline = Pipeline(config=config)
            pipeline.run()

            # Verify input file was copied with index "00"
            copied_file = output_dir / "00-test_input.md"
            assert copied_file.exists()
            assert copied_file.read_text() == "# Test Content"

    @patch("src.pipeline.Pipeline._initialize_phase")
    def test_start_from_phase_nonzero_does_not_copy_input_file(self, mock_initialize_phase):
        """Test that starting from phase > 0 does not copy the input file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Create fake output from phase 0
            phase_0_output = output_dir / "01-test_input Modernize_1.md"
            phase_0_output.write_text("# Test Content (modernized)")

            config = RunConfig(
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=input_file,
                phases=[
                    PhaseConfig(phase_type=PhaseType.MODERNIZE),
                    PhaseConfig(phase_type=PhaseType.EDIT),
                ],
                start_from_phase=1,
            )

            # Mock the phase
            mock_phase = MagicMock()
            mock_phase.name = "edit"
            mock_phase.output_file_path = output_dir / "02-test_input Edit_1.md"
            mock_initialize_phase.return_value = mock_phase

            pipeline = Pipeline(config=config)
            pipeline.run()

            # Verify input file was NOT copied (should not exist with index "00")
            copied_file = output_dir / "00-test_input.md"
            assert not copied_file.exists()

    def test_disabled_phases_distinct_from_skipped_phases(self):
        """Test that disabled phases are marked differently from skipped phases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Create fake output from previous phases
            phase_0_output = output_dir / "01-test_input Modernize_1.md"
            phase_0_output.write_text("# Test Content (modernized)")

            phase_1_output = output_dir / "02-test_input Edit_1.md"
            phase_1_output.write_text("# Test Content (edited)")

            config = RunConfig(
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=input_file,
                phases=[
                    PhaseConfig(phase_type=PhaseType.MODERNIZE),  # Will be skipped
                    PhaseConfig(phase_type=PhaseType.EDIT),  # Will be skipped
                    PhaseConfig(phase_type=PhaseType.FINAL, enabled=False),  # Disabled
                ],
                start_from_phase=2,
            )

            pipeline = Pipeline(config=config)
            pipeline.run()

            # Check metadata
            assert len(pipeline._phase_metadata) == 3
            # Phases 0 and 1 should be marked as skipped
            assert pipeline._phase_metadata[0]["reason"] == "skipped"
            assert pipeline._phase_metadata[1]["reason"] == "skipped"
            # Phase 2 should be marked as disabled
            assert pipeline._phase_metadata[2]["reason"] == "disabled"
