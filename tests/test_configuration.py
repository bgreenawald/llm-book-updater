"""
Configuration edge case tests for the LLM book updater.

These tests verify that the system handles invalid, missing, or malformed
configurations gracefully and provides appropriate error messages.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.core.pipeline import Pipeline
from src.models.config import PhaseConfig, PhaseType, RunConfig
from src.processing.post_processors import (
    PostProcessorChain,
    PreserveFStringTagsProcessor,
    RemoveXmlTagsProcessor,
)


class TestConfigurationValidation:
    """Tests for configuration validation and error handling."""

    def test_missing_input_file(self):
        """Test behavior when input file doesn't exist and pipeline tries to use it."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Reference non-existent input file
            non_existent_file = temp_path / "does_not_exist.md"
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Config creation should succeed (no validation at creation time)
            config = RunConfig(
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=non_existent_file,
                output_dir=output_dir,
                original_file=non_existent_file,
                phases=[],
            )

            # But using it should fail
            pipeline = Pipeline(config=config)
            with pytest.raises(Exception) as exc_info:
                pipeline._copy_input_file_to_output()

            # Should indicate file-related error
            assert (
                "exist" in str(exc_info.value).lower()
                or "found" in str(exc_info.value).lower()
                or "no such file" in str(exc_info.value).lower()
            )

    def test_invalid_output_directory(self):
        """Test behavior when output directory is invalid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create valid input file
            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")

            # Use a file as output directory (invalid)
            invalid_output_dir = temp_path / "not_a_directory.txt"
            invalid_output_dir.write_text("This is a file, not a directory")

            with pytest.raises(Exception) as exc_info:
                config = RunConfig(
                    book_id="test_book",
                    book_name="Test Book",
                    author_name="Test Author",
                    input_file=input_file,
                    output_dir=invalid_output_dir,
                    original_file=input_file,
                    phases=[],
                )
                pipeline = Pipeline(config=config)
                pipeline._copy_input_file_to_output()

            # Should indicate directory-related error
            assert "directory" in str(exc_info.value).lower() or "not a directory" in str(exc_info.value).lower()

    def test_empty_book_name(self):
        """Test behavior with empty or None book name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Test empty string - should be allowed at config level
            config = RunConfig(
                book_id="test_book",
                book_name="",  # Empty string
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=input_file,
                phases=[],
            )
            # Verify empty string is stored
            assert config.book_name == ""

            # Test None - Pydantic enforces type hints at runtime
            with pytest.raises(ValidationError):
                RunConfig(
                    book_id="test_book",
                    book_name=None,  # type: ignore[arg-type]
                    author_name="Test Author",
                    input_file=input_file,
                    output_dir=output_dir,
                    original_file=input_file,
                    phases=[],
                )

    def test_invalid_phase_configuration(self):
        """Test behavior with invalid phase configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Test invalid phase type
            with pytest.raises(ValidationError):
                RunConfig(
                    book_id="test_book",
                    book_name="Test Book",
                    author_name="Test Author",
                    input_file=input_file,
                    output_dir=output_dir,
                    original_file=input_file,
                    phases=[
                        PhaseConfig(
                            phase_type="INVALID_PHASE_TYPE",  # Invalid type
                            enabled=True,
                        )
                    ],
                )


class TestPostProcessorConfigurationEdgeCases:
    """Tests for post-processor configuration edge cases."""

    def test_empty_processor_chain(self):
        """Test behavior with empty post-processor chain."""
        chain = PostProcessorChain()

        # Should handle empty chain gracefully
        result = chain.process(original_block="Test content", llm_block="Modified content")
        assert result == "Modified content", "Empty chain should return input unchanged"

    def test_none_processor_config(self):
        """Test processors with None configuration."""
        # Test processor with None config
        processor = PreserveFStringTagsProcessor(config=None)

        original_block = "{preface}\nContent here\n{license}"
        llm_block = "Content here"

        result = processor.process(original_block=original_block, llm_block=llm_block)

        # Should use default configuration
        assert "{preface}" in result
        assert "{license}" in result

    def test_malformed_processor_config(self):
        """Test processors with malformed configuration."""
        # Test with invalid config structure
        malformed_config = {
            "tags_to_preserve": "not_a_list",  # Should be list, not string
        }

        # Processor creation succeeds, but should handle malformed config gracefully
        processor = PreserveFStringTagsProcessor(config=malformed_config)

        # Should either work by converting to list or handle gracefully
        try:
            result = processor.process(original_block="{preface}", llm_block="content")
            # If it works, verify it's a string result
            assert isinstance(result, str)
        except (TypeError, AttributeError) as e:
            # If it fails, should be due to string operations on non-list
            assert "str" in str(e).lower() or "list" in str(e).lower()

    def test_processor_with_unicode_config(self):
        """Test processors with Unicode characters in configuration."""
        unicode_config = {"tags_to_preserve": ["{préface}", "{许可证}", "{лицензия}"]}

        processor = PreserveFStringTagsProcessor(config=unicode_config)

        original_block = "{préface}\nContent\n{许可证}\nMore content\n{лицензия}"
        llm_block = "Content\nMore content"

        result = processor.process(original_block=original_block, llm_block=llm_block)

        # Should handle Unicode tags correctly
        assert "{préface}" in result
        assert "{许可证}" in result
        assert "{лицензия}" in result

    def test_processor_chain_with_failing_processor(self):
        """Test processor chain fails fast when one processor fails."""

        class FailingProcessor:
            def __init__(self):
                self.name = "failing_processor"

            def process(self, original_block, llm_block, **kwargs):
                raise Exception("Processor intentionally failed")

        class WorkingProcessor:
            def __init__(self):
                self.name = "working_processor"

            def process(self, original_block, llm_block, **kwargs):
                return llm_block.replace("old", "new")

        chain = PostProcessorChain()
        chain.add_processor(WorkingProcessor())
        chain.add_processor(FailingProcessor())
        chain.add_processor(WorkingProcessor())

        # Chain should fail fast when a processor raises an exception
        with pytest.raises(RuntimeError) as exc_info:
            chain.process(original_block="", llm_block="old content old")

        assert "Post-processing failed at processor failing_processor" in str(exc_info.value)

    def test_processor_with_extremely_large_config(self):
        """Test processor behavior with extremely large configuration."""
        # Create config with very large tag list
        large_tag_list = [f"{{tag_{i}}}" for i in range(10000)]
        large_config = {"tags_to_preserve": large_tag_list}

        processor = PreserveFStringTagsProcessor(config=large_config)

        # Should handle large config without crashing
        result = processor.process(original_block="{tag_0}", llm_block="content")

        assert "{tag_0}" in result

    def test_configuration_with_special_characters(self):
        """Test configuration with special characters and edge cases."""
        special_config = {
            "tags_to_preserve": [
                "{tag with spaces}",
                "{tag-with-dashes}",
                "{tag_with_underscores}",
                "{tag.with.dots}",
                "{tag@with@symbols}",
                "{'quotes'}",
                '{"double_quotes"}',
                "{[brackets]}",
                "{(parentheses)}",
            ]
        }

        processor = PreserveFStringTagsProcessor(config=special_config)

        original_block = "\n".join(special_config["tags_to_preserve"]) + "\nContent"
        llm_block = "Content"

        result = processor.process(original_block=original_block, llm_block=llm_block)

        # Should preserve all special character tags
        for tag in special_config["tags_to_preserve"]:
            assert tag in result, f"Tag {tag} was not preserved"


class TestErrorRecoveryAndGracefulDegradation:
    """Tests for error recovery and graceful degradation."""

    def test_partial_configuration_recovery(self):
        """Test system behavior when configuration is partially invalid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Valid config with one invalid phase
            try:
                config = RunConfig(
                    book_id="test_book",
                    book_name="Test Book",
                    author_name="Test Author",
                    input_file=input_file,
                    output_dir=output_dir,
                    original_file=input_file,
                    phases=[
                        PhaseConfig(
                            phase_type=PhaseType.MODERNIZE,
                            enabled=True,
                        ),
                        # This phase has invalid configuration but system should handle it
                        PhaseConfig(
                            phase_type=PhaseType.EDIT,
                            enabled=True,
                        ),
                    ],
                )

                pipeline = Pipeline(config=config)

                # Should be able to create pipeline even with some invalid phases
                assert pipeline is not None
                assert len(pipeline.config.phases) == 2

            except Exception as e:
                # If it fails, ensure error message is helpful
                assert "phase" in str(e).lower() or "config" in str(e).lower()

    def test_graceful_handling_of_missing_dependencies(self):
        """Test graceful handling when optional dependencies are missing."""
        # Mock a missing dependency
        with patch("src.processing.post_processors.re", None):
            # Should either work with degraded functionality or fail gracefully
            try:
                processor = RemoveXmlTagsProcessor()
                result = processor.process(original_block="", llm_block="<p>content</p>")
                # If it works, that's fine
                assert isinstance(result, str)
            except (ImportError, AttributeError) as e:
                # If it fails, should be due to missing re module
                error_str = str(e).lower()
                assert (
                    ("none" in error_str and "attribute" in error_str) or "module" in error_str or "import" in error_str
                )

    def test_configuration_validation_with_warnings(self):
        """Test that configuration warnings are properly handled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "test_input.md"
            input_file.write_text("# Test Content")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Configuration that might generate warnings but still be valid
            config = RunConfig(
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=input_file,
                output_dir=output_dir,
                original_file=input_file,
                phases=[
                    PhaseConfig(
                        phase_type=PhaseType.MODERNIZE,
                        enabled=True,
                    )
                ],
            )

            # Should create successfully despite potential warnings
            pipeline = Pipeline(config=config)
            assert pipeline is not None
