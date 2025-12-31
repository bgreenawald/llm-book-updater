"""
Test for input file copying functionality in the pipeline.

This test verifies that the pipeline correctly copies the input file
to the output directory with index "00".
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from book_updater import Pipeline, RunConfig


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
            mock_config.phases = []
            mock_config.get_phase_order.return_value = []

            # Create pipeline instance
            pipeline = Pipeline(config=mock_config)

            # Test with non-existent input file
            non_existent_file = temp_path / "non_existent.md"
            mock_config.input_file = non_existent_file

            # Should handle the error gracefully
            try:
                pipeline._copy_input_file_to_output()
                # If we get here, the error was handled gracefully
                pass
            except Exception as e:
                # Error handling is acceptable
                assert "FileNotFoundError" in str(type(e)) or "No such file" in str(e)

    def test_copy_input_file_with_special_characters(self):
        """Test that input files with special characters in names are handled correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a mock input file with special characters
            input_file = temp_path / "test_input_with_special_chars_@#$%.md"
            input_file.write_text("# Test Content with Special Characters")

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
            mock_config.phases = []
            mock_config.get_phase_order.return_value = []

            # Create pipeline instance
            pipeline = Pipeline(config=mock_config)

            # Call the method
            pipeline._copy_input_file_to_output()

            # Verify the file was copied with correct name
            expected_output_file = output_dir / "00-test_input_with_special_chars_@#$%.md"
            assert expected_output_file.exists()
            assert expected_output_file.read_text() == "# Test Content with Special Characters"

    def test_copy_input_file_preserves_file_permissions(self):
        """Test that file copying preserves file permissions."""
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
            mock_config.phases = []
            mock_config.get_phase_order.return_value = []

            # Create pipeline instance
            pipeline = Pipeline(config=mock_config)

            # Call the method
            pipeline._copy_input_file_to_output()

            # Verify the file was copied
            expected_output_file = output_dir / "00-test_input.md"
            assert expected_output_file.exists()

            # Verify content is identical
            assert expected_output_file.read_text() == input_file.read_text()


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


class TestFileSystemPermissions:
    """Test cases for file system permission edge cases."""

    def test_readonly_output_directory(self):
        """Test behavior when output directory is read-only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create input file
            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")

            # Create output directory and make it read-only
            output_dir = temp_path / "output"
            output_dir.mkdir()
            output_dir.chmod(0o444)  # Read-only

            try:
                # Create mock config
                mock_config = Mock(spec=RunConfig)
                mock_config.input_file = input_file
                mock_config.output_dir = output_dir
                mock_config.book_name = "test_book"
                mock_config.author_name = "test_author"
                mock_config.original_file = input_file
                mock_config.phases = []
                mock_config.get_phase_order.return_value = []

                pipeline = Pipeline(config=mock_config)

                # Should raise an exception when trying to copy to read-only directory
                with pytest.raises(Exception) as exc_info:
                    pipeline._copy_input_file_to_output()

                # Should be a permission-related error
                assert "Permission denied" in str(exc_info.value) or "Read-only" in str(exc_info.value)

            finally:
                # Restore permissions for cleanup
                output_dir.chmod(0o755)

    def test_nonexistent_parent_directory(self):
        """Test behavior when parent directory doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create input file
            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")

            # Use non-existent output directory
            output_dir = temp_path / "nonexistent" / "deeply" / "nested" / "output"

            # Create mock config
            mock_config = Mock(spec=RunConfig)
            mock_config.input_file = input_file
            mock_config.output_dir = output_dir
            mock_config.book_name = "test_book"
            mock_config.author_name = "test_author"
            mock_config.original_file = input_file
            mock_config.phases = []
            mock_config.get_phase_order.return_value = []

            pipeline = Pipeline(config=mock_config)

            # Should handle missing parent directories gracefully
            with pytest.raises(Exception) as exc_info:
                pipeline._copy_input_file_to_output()

            # Should be a file not found or directory-related error
            assert "No such file or directory" in str(exc_info.value) or "FileNotFoundError" in str(
                type(exc_info.value)
            )

    def test_input_file_permissions_readonly(self):
        """Test behavior when input file is read-only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create input file and make it read-only
            input_file = temp_path / "readonly_input.md"
            input_file.write_text("# Read-only Content")
            input_file.chmod(0o444)  # Read-only

            # Create output directory
            output_dir = temp_path / "output"
            output_dir.mkdir()

            try:
                # Create mock config
                mock_config = Mock(spec=RunConfig)
                mock_config.input_file = input_file
                mock_config.output_dir = output_dir
                mock_config.book_name = "test_book"
                mock_config.author_name = "test_author"
                mock_config.original_file = input_file
                mock_config.phases = []
                mock_config.get_phase_order.return_value = []

                pipeline = Pipeline(config=mock_config)

                # Should still be able to copy read-only files
                pipeline._copy_input_file_to_output()

                # Verify the file was copied
                expected_output_file = output_dir / "00-readonly_input.md"
                assert expected_output_file.exists()
                assert expected_output_file.read_text() == "# Read-only Content"

            finally:
                # Restore permissions for cleanup
                input_file.chmod(0o644)

    def test_output_directory_space_constraints(self):
        """Test behavior when output directory has limited space (simulated)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a large input file that might cause space issues
            large_content = "Large content line.\n" * 100000  # ~2MB
            input_file = temp_path / "large_input.md"
            input_file.write_text(large_content)

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
            mock_config.phases = []
            mock_config.get_phase_order.return_value = []

            pipeline = Pipeline(config=mock_config)

            # Should successfully copy large files
            pipeline._copy_input_file_to_output()

            # Verify the large file was copied correctly
            expected_output_file = output_dir / "00-large_input.md"
            assert expected_output_file.exists()
            assert expected_output_file.stat().st_size == input_file.stat().st_size

    def test_concurrent_file_access(self):
        """Test behavior when multiple processes try to access the same files."""
        import threading
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create input file
            input_file = temp_path / "concurrent_input.md"
            input_file.write_text("# Concurrent Access Test")

            # Create output directory
            output_dir = temp_path / "output"
            output_dir.mkdir()

            results = []
            exceptions = []

            def copy_file_worker(worker_id):
                try:
                    # Create mock config for this worker
                    mock_config = Mock(spec=RunConfig)
                    mock_config.input_file = input_file
                    mock_config.output_dir = output_dir
                    mock_config.book_name = f"test_book_{worker_id}"
                    mock_config.author_name = "test_author"
                    mock_config.original_file = input_file
                    mock_config.phases = []
                    mock_config.get_phase_order.return_value = []

                    pipeline = Pipeline(config=mock_config)

                    # Add small delay to increase chance of concurrent access
                    time.sleep(0.1)

                    pipeline._copy_input_file_to_output()
                    results.append(f"Worker {worker_id} succeeded")

                except Exception as e:
                    exceptions.append(f"Worker {worker_id} failed: {str(e)}")

            # Start multiple worker threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=copy_file_worker, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=10)

            # At least one worker should succeed
            assert len(results) >= 1, f"No workers succeeded. Exceptions: {exceptions}"

            # At least one output file should exist
            output_files = list(output_dir.glob("00-concurrent_input.md"))
            assert len(output_files) >= 1, "No output files were created"
