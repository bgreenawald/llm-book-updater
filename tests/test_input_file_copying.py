"""
Test for input file copying functionality in the pipeline.

This test verifies that the pipeline correctly copies the input file
to the output directory with index "00".
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.config import RunConfig
from src.pipeline import Pipeline


class TestInputFileCopying:
    """Test cases for input file copying functionality."""

    def test_copy_input_file_to_output(self):
        """Test that input file is copied to output directory with 00 index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a mock input file
            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content\n\nThis is test content.")

            # Create output directory
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Create mock config
            mock_config = Mock(spec=RunConfig)
            mock_config.input_file = input_file
            mock_config.output_dir = output_dir
            mock_config.book_name = "test_book"
            mock_config.author_name = "test_author"
            mock_config.original_file = input_file
            mock_config.length_reduction = [50]
            mock_config.phases = []
            mock_config.get_phase_order.return_value = []

            # Create pipeline instance
            pipeline = Pipeline(config=mock_config)

            # Call the method
            pipeline._copy_input_file_to_output()

            # Verify the file was copied
            expected_output_file = output_dir / "00-test_input.md"
            assert expected_output_file.exists()
            assert expected_output_file.read_text() == "# Test Content\n\nThis is test content."

    def test_copy_input_file_preserves_original_filename(self):
        """Test that the copied file preserves the original filename with 00 prefix."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a mock input file with a complex name
            input_file = temp_path / "complex_filename_with_underscores.md"
            input_file.write_text("# Test Content")

            # Create output directory
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Create mock config
            mock_config = Mock(spec=RunConfig)
            mock_config.input_file = input_file
            mock_config.output_dir = output_dir
            mock_config.book_name = "test_book"
            mock_config.author_name = "test_author"
            mock_config.original_file = input_file
            mock_config.length_reduction = [50]
            mock_config.phases = []
            mock_config.get_phase_order.return_value = []

            # Create pipeline instance
            pipeline = Pipeline(config=mock_config)

            # Call the method
            pipeline._copy_input_file_to_output()

            # Verify the filename is correct
            expected_output_file = output_dir / "00-complex_filename_with_underscores.md"
            assert expected_output_file.exists()

    def test_copy_input_file_handles_errors(self):
        """Test that errors during file copying are properly handled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a mock input file
            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")

            # Create output directory
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Create mock config
            mock_config = Mock(spec=RunConfig)
            mock_config.input_file = input_file
            mock_config.output_dir = output_dir
            mock_config.book_name = "test_book"
            mock_config.author_name = "test_author"
            mock_config.original_file = input_file
            mock_config.length_reduction = [50]
            mock_config.phases = []
            mock_config.get_phase_order.return_value = []

            # Create pipeline instance
            pipeline = Pipeline(config=mock_config)

            # Mock shutil.copy2 to raise an exception
            with patch("src.pipeline.shutil.copy2") as mock_copy:
                mock_copy.side_effect = Exception("Copy failed")

                # Call the method and expect it to raise an exception
                with pytest.raises(Exception, match="Copy failed"):
                    pipeline._copy_input_file_to_output()


class TestBuildScriptOriginalFileDetection:
    """Test cases for the build script's original file detection."""

    def test_find_original_file_in_output(self):
        """Test that the build script correctly finds 00-indexed files."""
        # Import the function from the build script
        import sys

        sys.path.append("books/the_federalist_papers")
        from build import find_original_file_in_output

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some test files
            (temp_path / "01-test.md").write_text("Phase 1")
            (temp_path / "02-test.md").write_text("Phase 2")
            (temp_path / "00-original.md").write_text("Original")

            # Test the function
            result = find_original_file_in_output(temp_path)
            assert result is not None
            assert result.name == "00-original.md"
            assert result.read_text() == "Original"

    def test_find_original_file_not_found(self):
        """Test that the function returns None when no 00-indexed file exists."""
        import sys

        sys.path.append("books/the_federalist_papers")
        from build import find_original_file_in_output

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some test files without 00-indexed file
            (temp_path / "01-test.md").write_text("Phase 1")
            (temp_path / "02-test.md").write_text("Phase 2")

            # Test the function
            result = find_original_file_in_output(temp_path)
            assert result is None

    def test_find_original_file_empty_directory(self):
        """Test that the function handles empty directories."""
        import sys

        sys.path.append("books/the_federalist_papers")
        from build import find_original_file_in_output

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test the function with empty directory
            result = find_original_file_in_output(temp_path)
            assert result is None

    def test_build_script_formats_original_file(self):
        """Test that the build script formats the original file with preface and license."""
        import sys

        sys.path.append("books/the_federalist_papers")
        from build import find_original_file_in_output, format_markdown_file

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test directory structure
            book_dir = temp_path / "test_book"
            book_dir.mkdir()
            output_dir = book_dir / "output"
            output_dir.mkdir()

            # Create test files
            original_file = output_dir / "00-test_input.md"
            original_file.write_text("# Original Content\n\n{preface}\n\n{license}")

            preface_file = book_dir / "preface.md"
            preface_file.write_text("This is the preface content.")

            license_file = book_dir / "license.md"
            license_file.write_text("This is the license content.")

            # Test that the find_original_file_in_output function works
            found_file = find_original_file_in_output(output_dir)
            assert found_file is not None
            assert found_file.exists()

            # Test formatting directly on the found file
            format_markdown_file(found_file, "This is the preface content.", "This is the license content.")

            # Verify the content was formatted
            content = found_file.read_text()
            assert "This is the preface content." in content
            assert "This is the license content." in content
            assert "{preface}" not in content
            assert "{license}" not in content
