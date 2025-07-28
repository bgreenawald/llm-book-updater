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
        from books.base_builder import BaseBookBuilder, BookConfig

        # Create a simple test builder class
        class TestBuilder(BaseBookBuilder):
            def get_source_files(self):
                return {}

            def get_original_file(self):
                # Look for files starting with "00-" in the output directory
                if not self.config.source_output_dir.exists():
                    return None

                for file_path in self.config.source_output_dir.glob("00-*"):
                    if file_path.is_file():
                        return file_path

                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some test files
            (temp_path / "01-test.md").write_text("Phase 1")
            (temp_path / "02-test.md").write_text("Phase 2")
            (temp_path / "00-original.md").write_text("Original")

            # Create a config and builder to test the logic
            config = BookConfig(
                name="test_book",
                version="1.0.0",
                title="Test Book",
                author="Test Author",
            )
            # Override the source_output_dir to point to our test directory
            config.source_output_dir = temp_path

            builder = TestBuilder(config=config)

            # Test the function
            result = builder.get_original_file()
            assert result is not None
            assert result.name == "00-original.md"
            assert result.read_text() == "Original"

    def test_find_original_file_not_found(self):
        """Test that the function returns None when no 00-indexed file exists."""
        from books.base_builder import BaseBookBuilder, BookConfig

        # Create a simple test builder class
        class TestBuilder(BaseBookBuilder):
            def get_source_files(self):
                return {}

            def get_original_file(self):
                # Look for files starting with "00-" in the output directory
                if not self.config.source_output_dir.exists():
                    return None

                for file_path in self.config.source_output_dir.glob("00-*"):
                    if file_path.is_file():
                        return file_path

                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some test files without 00-indexed file
            (temp_path / "01-test.md").write_text("Phase 1")
            (temp_path / "02-test.md").write_text("Phase 2")

            # Create a config and builder to test the logic
            config = BookConfig(
                name="test_book",
                version="1.0.0",
                title="Test Book",
                author="Test Author",
            )
            # Override the source_output_dir to point to our test directory
            config.source_output_dir = temp_path

            builder = TestBuilder(config=config)

            # Test the function
            result = builder.get_original_file()
            assert result is None

    def test_find_original_file_empty_directory(self):
        """Test that the function handles empty directories."""
        from books.base_builder import BaseBookBuilder, BookConfig

        # Create a simple test builder class
        class TestBuilder(BaseBookBuilder):
            def get_source_files(self):
                return {}

            def get_original_file(self):
                # Look for files starting with "00-" in the output directory
                if not self.config.source_output_dir.exists():
                    return None

                for file_path in self.config.source_output_dir.glob("00-*"):
                    if file_path.is_file():
                        return file_path

                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a config and builder to test the logic
            config = BookConfig(
                name="test_book",
                version="1.0.0",
                title="Test Book",
                author="Test Author",
            )
            # Override the source_output_dir to point to our test directory
            config.source_output_dir = temp_path

            builder = TestBuilder(config=config)

            # Test the function with empty directory
            result = builder.get_original_file()
            assert result is None

    def test_build_script_formats_original_file(self):
        """Test that the build script formats the original file with preface and license."""
        from books.base_builder import BaseBookBuilder, BookConfig

        # Create a simple test builder class
        class TestBuilder(BaseBookBuilder):
            def get_source_files(self):
                return {}

            def get_original_file(self):
                # Look for files starting with "00-" in the output directory
                if not self.config.source_output_dir.exists():
                    return None

                for file_path in self.config.source_output_dir.glob("00-*"):
                    if file_path.is_file():
                        return file_path

                return None

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

            # Create a config and builder to test the logic
            config = BookConfig(
                name="test_book",
                version="1.0.0",
                title="Test Book",
                author="Test Author",
            )
            # Override the source_output_dir to point to our test directory
            config.source_output_dir = output_dir
            config.preface_md = preface_file
            config.license_md = license_file

            builder = TestBuilder(config=config)

            # Test that the get_original_file function works
            found_file = builder.get_original_file()
            assert found_file is not None
            assert found_file.exists()

            # Test formatting directly on the found file using the builder's method
            builder.format_markdown_file(
                found_file, "This is the preface content.", "This is the license content.", "1.0.0"
            )

            # Verify the content was formatted
            content = found_file.read_text()
            assert "This is the preface content." in content
            assert "This is the license content." in content
            assert "{preface}" not in content
            assert "{license}" not in content
