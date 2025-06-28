import pytest

from src.post_processors import (
    NoNewHeadersPostProcessor,
    PostProcessorChain,
    RemoveTrailingWhitespaceProcessor,
    RemoveXmlTagsProcessor,
    RevertRemovedBlockLines,
)

# ============================================================================
# NoNewHeadersPostProcessor Tests
# ============================================================================


@pytest.fixture
def no_new_headers_processor():
    return NoNewHeadersPostProcessor()


class TestNoNewHeadersPostProcessor:
    """Test suite for NoNewHeadersPostProcessor."""

    def test_revert_converted_header(self, no_new_headers_processor):
        """Test reverting a line that was converted to a header."""
        original_block = "This is a test line."
        llm_block = "# This is a test line."
        expected_output = "This is a test line."
        result = no_new_headers_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_remove_new_header(self, no_new_headers_processor):
        """Test removing an entirely new header."""
        original_block = "This is a test line."
        llm_block = "This is a test line.\n# This is a new header."
        expected_output = "This is a test line."
        result = no_new_headers_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_no_change_when_header_exists(self, no_new_headers_processor):
        """Test that existing headers are preserved."""
        original_block = "# This is a test line."
        llm_block = "# This is a test line."
        expected_output = "# This is a test line."
        result = no_new_headers_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_mixed_changes(self, no_new_headers_processor):
        """Test mixed scenarios with converted and new headers."""
        original_block = "First line.\nSecond line that will be converted.\n# An existing header."
        llm_block = (
            "First line.\n# Second line that will be converted.\n# An existing header.\n## A completely new header."
        )
        expected_output = "First line.\nSecond line that will be converted.\n# An existing header."
        result = no_new_headers_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_empty_input(self, no_new_headers_processor):
        """Test with empty input."""
        original_block = ""
        llm_block = ""
        expected_output = ""
        result = no_new_headers_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_llm_block_empty(self, no_new_headers_processor):
        """Test when LLM block is empty."""
        original_block = "Some content."
        llm_block = ""
        expected_output = ""
        result = no_new_headers_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_multiple_header_levels(self, no_new_headers_processor):
        """Test handling of different header levels."""
        original_block = "Regular text.\nSubsection text."
        llm_block = "Regular text.\n# Subsection text.\n## New subsection."
        expected_output = "Regular text.\nSubsection text."
        result = no_new_headers_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_header_with_extra_whitespace(self, no_new_headers_processor):
        """Test headers with extra whitespace."""
        original_block = "  This is a line with spaces  "
        llm_block = "# This is a line with spaces"
        expected_output = "  This is a line with spaces  "
        result = no_new_headers_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_multiple_converted_headers(self, no_new_headers_processor):
        """Test multiple lines converted to headers."""
        original_block = "Line 1.\nLine 2.\nLine 3."
        llm_block = "# Line 1.\n# Line 2.\nLine 3."
        expected_output = "Line 1.\nLine 2.\nLine 3."
        result = no_new_headers_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_header_with_special_characters(self, no_new_headers_processor):
        """Test headers with special characters."""
        original_block = "Line with @#$%^&*() characters"
        llm_block = "# Line with @#$%^&*() characters"
        expected_output = "Line with @#$%^&*() characters"
        result = no_new_headers_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_preserve_existing_headers_in_order(self, no_new_headers_processor):
        """Test that existing headers maintain their order."""
        original_block = "# Header 1\n# Header 2\n# Header 3"
        llm_block = "# Header 1\n# Header 2\n# Header 3"
        expected_output = "# Header 1\n# Header 2\n# Header 3"
        result = no_new_headers_processor.process(original_block, llm_block)
        assert result == expected_output


# ============================================================================
# RevertRemovedBlockLines Tests
# ============================================================================


@pytest.fixture
def revert_removed_block_lines_processor():
    return RevertRemovedBlockLines()


class TestRevertRemovedBlockLines:
    """Test suite for RevertRemovedBlockLines."""

    def test_restore_single_removed_block_line(self, revert_removed_block_lines_processor):
        """Test restoring a single removed block quote line."""
        original_block = "First line.\n> This is a block quote.\nSecond line."
        llm_block = "First line.\nSecond line."
        expected_output = "First line.\n> This is a block quote.\nSecond line."
        result = revert_removed_block_lines_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_restore_multiple_removed_block_lines(self, revert_removed_block_lines_processor):
        """Test restoring multiple consecutive block quote lines."""
        original_block = "First line.\n> Quote 1.\n> Quote 2.\nSecond line."
        llm_block = "First line.\nSecond line."
        expected_output = "First line.\n> Quote 1.\n> Quote 2.\nSecond line."
        result = revert_removed_block_lines_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_no_block_lines_removed(self, revert_removed_block_lines_processor):
        """Test when no block lines were removed."""
        original_block = "First line.\n> Quote 1.\nSecond line."
        llm_block = "First line.\n> Quote 1.\nSecond line."
        expected_output = "First line.\n> Quote 1.\nSecond line."
        result = revert_removed_block_lines_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_regular_lines_removed(self, revert_removed_block_lines_processor):
        """Test when regular lines are removed (should not restore)."""
        original_block = "First line.\nSecond line.\n> Quote 1."
        llm_block = "> Quote 1."
        expected_output = "> Quote 1."
        result = revert_removed_block_lines_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_mixed_content_with_removed_block_lines(self, revert_removed_block_lines_processor):
        """Test mixed content with block lines removed."""
        original_block = "Intro.\n> Important quote.\nSome text.\n> Another quote.\nOutro."
        llm_block = "Intro.\nSome text.\nOutro."
        expected_output = "Intro.\n> Important quote.\nSome text.\n> Another quote.\nOutro."
        result = revert_removed_block_lines_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_restore_consecutive_removed_block_lines(self, revert_removed_block_lines_processor):
        """Test restoring consecutive block lines."""
        original_block = "Line 1\n> Quote 1\n> Quote 2\n> Quote 3\nLine 2"
        llm_block = "Line 1\nLine 2"
        expected_output = "Line 1\n> Quote 1\n> Quote 2\n> Quote 3\nLine 2"
        result = revert_removed_block_lines_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_restore_removed_first_line_block_comment(self, revert_removed_block_lines_processor):
        """Test restoring block quote at the beginning."""
        original_block = "> This is the first line.\nSecond line."
        llm_block = "Second line."
        expected_output = "> This is the first line.\nSecond line."
        result = revert_removed_block_lines_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_restore_removed_last_line_block_comment(self, revert_removed_block_lines_processor):
        """Test restoring block quote at the end."""
        original_block = "First line.\n> This is the last line."
        llm_block = "First line."
        expected_output = "First line.\n> This is the last line."
        result = revert_removed_block_lines_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_mixed_input_empty_llm_output_restores_only_blocklines(self, revert_removed_block_lines_processor):
        """Test when LLM output is empty, only restore block lines."""
        original_block = "This is a regular line.\n> This is a block quote.\nAnother regular line."
        llm_block = ""
        expected_output = "> This is a block quote."
        result = revert_removed_block_lines_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_block_lines_with_whitespace(self, revert_removed_block_lines_processor):
        """Test block lines with leading/trailing whitespace."""
        original_block = "Line 1\n>  Block quote with spaces  \nLine 2"
        llm_block = "Line 1\nLine 2"
        expected_output = "Line 1\n>  Block quote with spaces  \nLine 2"
        result = revert_removed_block_lines_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_multiple_blocks_removed(self, revert_removed_block_lines_processor):
        """Test when multiple separate block quote sections are removed."""
        original_block = "Intro.\n> Quote 1.\nMiddle.\n> Quote 2.\nOutro."
        llm_block = "Intro.\nMiddle.\nOutro."
        expected_output = "Intro.\n> Quote 1.\nMiddle.\n> Quote 2.\nOutro."
        result = revert_removed_block_lines_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_block_lines_mixed_with_regular_quotes(self, revert_removed_block_lines_processor):
        """Test block lines mixed with regular quote lines."""
        original_block = "Text.\n> Block quote.\n> Regular quote.\n> Another block.\nMore text."
        llm_block = "Text.\n> Regular quote.\nMore text."
        expected_output = "Text.\n> Block quote.\n> Regular quote.\n> Another block.\nMore text."
        result = revert_removed_block_lines_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_empty_original_block(self, revert_removed_block_lines_processor):
        """Test with empty original block."""
        original_block = ""
        llm_block = "Some content."
        expected_output = "Some content."
        result = revert_removed_block_lines_processor.process(original_block, llm_block)
        assert result == expected_output

    def test_both_blocks_empty(self, revert_removed_block_lines_processor):
        """Test with both blocks empty."""
        original_block = ""
        llm_block = ""
        expected_output = ""
        result = revert_removed_block_lines_processor.process(original_block, llm_block)
        assert result == expected_output


# ============================================================================
# PostProcessorChain Tests
# ============================================================================


@pytest.fixture
def post_processor_chain():
    return PostProcessorChain()


@pytest.fixture
def sample_processors():
    return [
        RemoveXmlTagsProcessor(),
        RemoveTrailingWhitespaceProcessor(),
    ]


class TestPostProcessorChain:
    """Test suite for PostProcessorChain."""

    def test_empty_chain(self, post_processor_chain):
        """Test chain with no processors."""
        original_block = "Original content."
        llm_block = "LLM content."
        result = post_processor_chain.process(original_block, llm_block)
        assert result == llm_block

    def test_single_processor(self, post_processor_chain):
        """Test chain with single processor."""
        processor = RemoveXmlTagsProcessor()
        post_processor_chain.add_processor(processor)

        original_block = "Original content."
        llm_block = "<p>Some text.</p>"
        expected_output = "Some text."
        result = post_processor_chain.process(original_block, llm_block)
        assert result == expected_output

    def test_multiple_processors(self, post_processor_chain):
        """Test chain with multiple processors."""
        post_processor_chain.add_processor(RemoveXmlTagsProcessor())
        post_processor_chain.add_processor(RemoveTrailingWhitespaceProcessor())

        original_block = "Original content."
        llm_block = "<p>Some text.  </p>\n<br>\n<item>Another item</item>"
        expected_output = "Some text.\n<br>\nAnother item"
        result = post_processor_chain.process(original_block, llm_block)
        assert result == expected_output

    def test_chain_initialization_with_processors(self, sample_processors):
        """Test chain initialization with processors."""
        chain = PostProcessorChain(sample_processors)
        assert len(chain) == 2

    def test_add_processor(self, post_processor_chain):
        """Test adding processors to chain."""
        assert len(post_processor_chain) == 0

        processor = RemoveXmlTagsProcessor()
        post_processor_chain.add_processor(processor)
        assert len(post_processor_chain) == 1

    def test_chain_processing_order(self, post_processor_chain):
        """Test that processors are applied in correct order."""

        # Create a custom processor that adds markers to track order
        class OrderTracker:
            def __init__(self, name):
                self.name = name
                self.order = []

            def process(self, original_block, llm_block, **kwargs):
                self.order.append(self.name)
                return f"{llm_block}_{self.name}"

        tracker1 = OrderTracker("first")
        tracker2 = OrderTracker("second")

        post_processor_chain.add_processor(tracker1)
        post_processor_chain.add_processor(tracker2)

        original_block = "Original"
        llm_block = "LLM"
        result = post_processor_chain.process(original_block, llm_block)

        assert result == "LLM_first_second"
        assert tracker1.order == ["first"]
        assert tracker2.order == ["second"]

    def test_chain_with_error_handling(self, post_processor_chain):
        """Test chain continues processing when a processor fails."""

        class FailingProcessor:
            def __init__(self, name):
                self.name = name

            def process(self, original_block, llm_block, **kwargs):
                raise Exception(f"Error in {self.name}")

        class WorkingProcessor:
            def __init__(self, name):
                self.name = name

            def process(self, original_block, llm_block, **kwargs):
                return f"{llm_block}_{self.name}"

        post_processor_chain.add_processor(WorkingProcessor("working"))
        post_processor_chain.add_processor(FailingProcessor("failing"))
        post_processor_chain.add_processor(WorkingProcessor("working2"))

        original_block = "Original"
        llm_block = "LLM"
        result = post_processor_chain.process(original_block, llm_block)

        # Should continue processing and apply working2
        assert result == "LLM_working_working2"

    def test_chain_string_representation(self, post_processor_chain):
        """Test string representation of chain."""
        post_processor_chain.add_processor(RemoveXmlTagsProcessor())
        post_processor_chain.add_processor(RemoveTrailingWhitespaceProcessor())

        expected = "PostProcessorChain(['remove_xml_tags', 'remove_trailing_whitespace'])"
        assert str(post_processor_chain) == expected

    def test_chain_length(self, post_processor_chain):
        """Test chain length property."""
        assert len(post_processor_chain) == 0
        post_processor_chain.add_processor(RemoveXmlTagsProcessor())
        assert len(post_processor_chain) == 1
        post_processor_chain.add_processor(RemoveTrailingWhitespaceProcessor())
        assert len(post_processor_chain) == 2

    def test_chain_with_kwargs(self, post_processor_chain):
        """Test chain passes kwargs to processors."""

        class KwargsProcessor:
            def __init__(self):
                self.name = "kwargs_test"

            def process(self, original_block, llm_block, **kwargs):
                return f"{llm_block}_{kwargs.get('test_param', 'default')}"

        post_processor_chain.add_processor(KwargsProcessor())

        original_block = "Original"
        llm_block = "LLM"
        result = post_processor_chain.process(original_block, llm_block, test_param="custom")
        assert result == "LLM_custom"

    def test_chain_with_empty_llm_block(self, post_processor_chain):
        """Test chain with empty LLM block."""
        post_processor_chain.add_processor(RemoveXmlTagsProcessor())

        original_block = "Original content."
        llm_block = ""
        result = post_processor_chain.process(original_block, llm_block)
        assert result == ""

    def test_chain_with_empty_original_block(self, post_processor_chain):
        """Test chain with empty original block."""
        post_processor_chain.add_processor(RemoveXmlTagsProcessor())

        original_block = ""
        llm_block = "<p>Some content.</p>"
        expected_output = "Some content."
        result = post_processor_chain.process(original_block, llm_block)
        assert result == expected_output

    def test_chain_with_complex_processors(self, post_processor_chain):
        """Test chain with complex processors that modify content significantly."""
        post_processor_chain.add_processor(RemoveXmlTagsProcessor())
        post_processor_chain.add_processor(RemoveTrailingWhitespaceProcessor())
        post_processor_chain.add_processor(NoNewHeadersPostProcessor())

        original_block = "Regular text.\nSubsection text."
        llm_block = "<p>Regular text.</p>\n# Subsection text.\n## New header.\n<br>\n  Trailing spaces  "

        expected_output = "Regular text.\nSubsection text.\n<br>\n  Trailing spaces"
        result = post_processor_chain.process(original_block, llm_block)
        assert result == expected_output
