import pytest

from src.post_processors import (
    NoNewHeadersPostProcessor,
    PostProcessor,
    PostProcessorChain,
    PreserveFStringTagsProcessor,
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
        original_block = "# Header\n\nContent here."
        llm_block = "# Header\n\n# New Header\n\nContent here."

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        assert "# New Header" not in result
        assert "# Header" in result

    def test_remove_new_header(self, no_new_headers_processor):
        """Test removing an entirely new header."""
        original_block = "Content here."
        llm_block = "# New Header\n\nContent here."

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        assert "# New Header" not in result
        assert "Content here." in result

    def test_no_change_when_header_exists(self, no_new_headers_processor):
        """Test that existing headers are preserved."""
        original_block = "# Header\n\nContent here."
        llm_block = "# Header\n\nContent here."

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        assert result == llm_block

    def test_mixed_changes(self, no_new_headers_processor):
        """Test mixed scenarios with converted and new headers."""
        original_block = "# Header\n\nContent here."
        llm_block = "# Header\n\n# New Header\n\nContent here.\n\n# Another New Header\n\nMore content."

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        assert "# New Header" not in result
        assert "# Another New Header" not in result
        assert "# Header" in result
        assert "Content here." in result
        assert "More content." in result

    def test_empty_input(self, no_new_headers_processor):
        """Test with empty input."""
        original_block = ""
        llm_block = "# New Header\n\nContent here."

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        assert "# New Header" not in result
        assert "Content here." in result

    def test_llm_block_empty(self, no_new_headers_processor):
        """Test when LLM block is empty."""
        original_block = "# Header\n\nContent here."
        llm_block = ""

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        assert result == ""

    def test_multiple_header_levels(self, no_new_headers_processor):
        """Test handling of different header levels."""
        original_block = "Content here."
        llm_block = "# Header 1\n\n## Header 2\n\n### Header 3\n\nContent here."

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        assert "# Header 1" not in result
        assert "## Header 2" not in result
        assert "### Header 3" not in result
        assert "Content here." in result

    def test_header_with_extra_whitespace(self, no_new_headers_processor):
        """Test headers with extra whitespace."""
        original_block = "Content here."
        llm_block = "   #   Header   \n\nContent here."

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        # The processor does not strip leading whitespace, so '#   Header' is still present
        assert "#   Header" in result
        assert "Content here." in result

    def test_multiple_converted_headers(self, no_new_headers_processor):
        """Test multiple lines converted to headers."""
        original_block = "Line 1\nLine 2\nLine 3"
        llm_block = "# Line 1\n\n# Line 2\n\n# Line 3"

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        assert "# Line 1" not in result
        assert "# Line 2" not in result
        assert "# Line 3" not in result
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_header_with_special_characters(self, no_new_headers_processor):
        """Test headers with special characters."""
        original_block = "Content here."
        llm_block = "# Header with **bold** and *italic*\n\nContent here."

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        assert "# Header with **bold** and *italic*" not in result
        assert "Content here." in result

    def test_preserve_existing_headers_in_order(self, no_new_headers_processor):
        """Test that existing headers are preserved in their original order."""
        original_block = "# Header 1\n\n## Header 2\n\nContent here."
        llm_block = "# Header 1\n\n## Header 2\n\n# New Header\n\nContent here."

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        # Check that existing headers are preserved in order
        lines = result.split("\n")
        header_lines = [line for line in lines if line.strip().startswith("#")]
        assert header_lines == ["# Header 1", "## Header 2"]
        assert "# New Header" not in result

    def test_multiple_new_headers_removed(self, no_new_headers_processor):
        """Test that multiple new headers are properly removed."""
        original_block = "Content here."
        llm_block = "# New Header 1\n\n# New Header 2\n\nContent here."

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        assert "# New Header 1" not in result
        assert "# New Header 2" not in result
        assert "Content here." in result

    def test_header_with_special_characters_in_content(self, no_new_headers_processor):
        """Test headers with special characters in content are handled correctly."""
        original_block = "Content here."
        llm_block = "# Header with [brackets] and (parentheses)\n\nContent here."

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        assert "# Header with [brackets] and (parentheses)" not in result
        assert "Content here." in result

    def test_multiple_new_headers_with_content_between(self, no_new_headers_processor):
        """Test multiple new headers with content between them."""
        original_block = "# Original Header\n\nContent here."
        llm_block = "# Original Header\n\n# New Header 1\n\nSome content.\n\n# New Header 2\n\nMore content."

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        assert "# New Header 1" not in result
        assert "# New Header 2" not in result
        assert "# Original Header" in result
        assert "Some content." in result
        assert "More content." in result

    def test_headers_with_leading_whitespace_not_removed(self, no_new_headers_processor):
        """Test that headers with leading whitespace are not removed (current behavior)."""
        original_block = "Content here."
        llm_block = "  # Header with leading spaces\n\n\t# Header with leading tab\n\nContent here."

        result = no_new_headers_processor.process(original_block=original_block, llm_block=llm_block)

        # Current implementation doesn't match headers with leading whitespace
        # So these should remain in the output
        assert "  # Header with leading spaces" in result
        assert "\t# Header with leading tab" in result
        assert "Content here." in result


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
        original_block = "> Block line 1\nRegular line\n> Block line 2"
        llm_block = "Regular line"

        result = revert_removed_block_lines_processor.process(original_block=original_block, llm_block=llm_block)

        assert "> Block line 1" in result
        assert "> Block line 2" in result
        assert "Regular line" in result

    def test_restore_multiple_removed_block_lines(self, revert_removed_block_lines_processor):
        """Test restoring multiple consecutive block quote lines."""
        original_block = "> Block line 1\n> Block line 2\nRegular line\n> Block line 3"
        llm_block = "Regular line"

        result = revert_removed_block_lines_processor.process(original_block=original_block, llm_block=llm_block)

        assert "> Block line 1" in result
        assert "> Block line 2" in result
        assert "> Block line 3" in result
        assert "Regular line" in result

    def test_no_block_lines_removed(self, revert_removed_block_lines_processor):
        """Test when no block lines were removed."""
        original_block = "Regular line 1\nRegular line 2"
        llm_block = "Regular line 1\nRegular line 2"

        result = revert_removed_block_lines_processor.process(original_block=original_block, llm_block=llm_block)

        assert result == llm_block

    def test_regular_lines_removed(self, revert_removed_block_lines_processor):
        """Test when regular lines are removed (should not restore)."""
        original_block = "Regular line 1\nRegular line 2\nRegular line 3"
        llm_block = "Regular line 1\nRegular line 3"

        result = revert_removed_block_lines_processor.process(original_block=original_block, llm_block=llm_block)

        # Should not restore regular lines
        assert "Regular line 2" not in result
        assert "Regular line 1" in result
        assert "Regular line 3" in result

    def test_mixed_content_with_removed_block_lines(self, revert_removed_block_lines_processor):
        """Test mixed content with block lines removed."""
        original_block = "Regular line 1\n> Block line 1\nRegular line 2\n> Block line 2"
        llm_block = "Regular line 1\nRegular line 2"

        result = revert_removed_block_lines_processor.process(original_block=original_block, llm_block=llm_block)

        assert "> Block line 1" in result
        assert "> Block line 2" in result
        assert "Regular line 1" in result
        assert "Regular line 2" in result

    def test_restore_consecutive_removed_block_lines(self, revert_removed_block_lines_processor):
        """Test restoring consecutive block lines."""
        original_block = "> Block line 1\n> Block line 2\n> Block line 3\nRegular line"
        llm_block = "Regular line"

        result = revert_removed_block_lines_processor.process(original_block=original_block, llm_block=llm_block)

        assert "> Block line 1" in result
        assert "> Block line 2" in result
        assert "> Block line 3" in result
        assert "Regular line" in result

    def test_restore_removed_first_line_block_comment(self, revert_removed_block_lines_processor):
        """Test restoring block quote at the beginning."""
        original_block = "> Block line 1\nRegular line 1\nRegular line 2"
        llm_block = "Regular line 1\nRegular line 2"

        result = revert_removed_block_lines_processor.process(original_block=original_block, llm_block=llm_block)

        assert "> Block line 1" in result
        assert "Regular line 1" in result
        assert "Regular line 2" in result

    def test_restore_removed_last_line_block_comment(self, revert_removed_block_lines_processor):
        """Test restoring block quote at the end."""
        original_block = "Regular line 1\nRegular line 2\n> Block line 1"
        llm_block = "Regular line 1\nRegular line 2"

        result = revert_removed_block_lines_processor.process(original_block=original_block, llm_block=llm_block)

        assert "> Block line 1" in result
        assert "Regular line 1" in result
        assert "Regular line 2" in result

    def test_mixed_input_empty_llm_output_restores_only_blocklines(self, revert_removed_block_lines_processor):
        """Test when LLM output is empty, only restore block lines."""
        original_block = "Regular line 1\n> Block line 1\nRegular line 2\n> Block line 2"
        llm_block = ""

        result = revert_removed_block_lines_processor.process(original_block=original_block, llm_block=llm_block)

        assert "> Block line 1" in result
        assert "> Block line 2" in result
        assert "Regular line 1" not in result
        assert "Regular line 2" not in result

    def test_block_lines_with_whitespace(self, revert_removed_block_lines_processor):
        """Test block lines with leading/trailing whitespace."""
        original_block = ">   Block line with spaces  \nRegular line"
        llm_block = "Regular line"

        result = revert_removed_block_lines_processor.process(original_block=original_block, llm_block=llm_block)

        assert ">   Block line with spaces  " in result
        assert "Regular line" in result

    def test_multiple_blocks_removed(self, revert_removed_block_lines_processor):
        """Test when multiple separate block quote sections are removed."""
        original_block = "> Block 1 line 1\n> Block 1 line 2\nRegular line\n> Block 2 line 1\n> Block 2 line 2"
        llm_block = "Regular line"

        result = revert_removed_block_lines_processor.process(original_block=original_block, llm_block=llm_block)

        assert "> Block 1 line 1" in result
        assert "> Block 1 line 2" in result
        assert "> Block 2 line 1" in result
        assert "> Block 2 line 2" in result
        assert "Regular line" in result

    def test_block_lines_mixed_with_regular_quotes(self):
        """Test block lines mixed with regular quotes and regular lines."""
        processor = RevertRemovedBlockLines()
        original_block = "> Block line 1\n> Regular quote\nRegular line\n> Block line 3"
        llm_block = "> Regular quote\nRegular line\n> Block line 3"

        result = processor.process(original_block=original_block, llm_block=llm_block)

        # Only lines starting with '> ' that are deleted should be restored
        assert "> Block line 1" in result
        assert "> Regular quote" in result
        assert "Regular line" in result
        assert "> Block line 3" in result

    def test_empty_original_block(self, revert_removed_block_lines_processor):
        """Test with empty original block."""
        original_block = ""
        llm_block = "Some content"

        result = revert_removed_block_lines_processor.process(original_block=original_block, llm_block=llm_block)

        assert result == llm_block

    def test_both_blocks_empty(self, revert_removed_block_lines_processor):
        """Test with both blocks empty."""
        original_block = ""
        llm_block = ""

        result = revert_removed_block_lines_processor.process(original_block=original_block, llm_block=llm_block)

        assert result == ""


# ============================================================================
# PreserveFStringTagsProcessor Tests
# ============================================================================


@pytest.fixture
def preserve_fstring_tags_processor():
    return PreserveFStringTagsProcessor()


@pytest.fixture
def custom_tags_processor():
    return PreserveFStringTagsProcessor(config={"tags_to_preserve": ["{custom}", "{special}"]})


class TestPreserveFStringTagsProcessor:
    """Test suite for PreserveFStringTagsProcessor."""

    def test_no_tags_in_original(self, preserve_fstring_tags_processor):
        """Test when no special tags are present in the original."""
        original_block = "This is a regular paragraph with no special tags."
        llm_block = "This is the processed content."

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        assert result == llm_block

    def test_tags_preserved_in_llm(self, preserve_fstring_tags_processor):
        """Test when tags are already preserved in the LLM output."""
        original_block = "This is a paragraph with {preface} and {license} tags."
        llm_block = "This is the processed content with {preface} and {license} tags."

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        assert result == llm_block

    def test_restore_single_tag(self, preserve_fstring_tags_processor):
        """Test restoring a single missing tag."""
        original_block = "{preface}"
        llm_block = "This is the processed content."

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "This is the processed content.\n{preface}"
        assert result == expected

    def test_restore_multiple_tags(self, preserve_fstring_tags_processor):
        """Test restoring multiple missing tags."""
        original_block = "{preface}\n{license}"
        llm_block = "This is the processed content."

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "This is the processed content.\n{preface}\n{license}"
        assert result == expected

    def test_restore_tag_at_same_position(self, preserve_fstring_tags_processor):
        """Test restoring a tag at the same position as in the original."""
        original_block = "Some content\n{preface}\nMore content"
        llm_block = "Some content\nMore content"

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "Some content\n{preface}\nMore content"
        assert result == expected

    def test_restore_tag_at_end_of_line(self, preserve_fstring_tags_processor):
        """Test restoring a tag at the end of content."""
        original_block = "Some content\n{preface}"
        llm_block = "Some content"

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "Some content\n{preface}"
        assert result == expected

    def test_restore_tag_at_beginning_of_line(self, preserve_fstring_tags_processor):
        """Test restoring a tag at the beginning of content."""
        original_block = "{preface}\nSome content"
        llm_block = "Some content"

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "{preface}\nSome content"
        assert result == expected

    def test_restore_tags_in_multiple_lines(self, preserve_fstring_tags_processor):
        """Test restoring tags in multiple lines."""
        original_block = "Line 1\n{preface}\nLine 2\n{license}\nLine 3"
        llm_block = "Line 1\nLine 2\nLine 3"

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "Line 1\n{preface}\nLine 2\n{license}\nLine 3"
        assert result == expected

    def test_restore_tags_with_context_matching(self, preserve_fstring_tags_processor):
        """Test restoring tags when similar context is found."""
        original_block = "Some content\n{preface}\nMore content"
        llm_block = "Some content\nMore content"

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "Some content\n{preface}\nMore content"
        assert result == expected

    def test_restore_tags_when_no_similar_line_found(self, preserve_fstring_tags_processor):
        """Test restoring tags when no similar line is found."""
        original_block = "Original line\n{preface}"
        llm_block = "Completely different content."

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "Completely different content.\n{preface}"
        assert result == expected

    def test_restore_tags_with_context_before(self, preserve_fstring_tags_processor):
        """Test restoring tags using context before the tag."""
        original_block = "Some content\n{preface}"
        llm_block = "Some content"

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "Some content\n{preface}"
        assert result == expected

    def test_restore_tags_with_context_after(self, preserve_fstring_tags_processor):
        """Test restoring tags using context after the tag."""
        original_block = "Some content\n{preface}\nMore content"
        llm_block = "Some content\nMore content"

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "Some content\n{preface}\nMore content"
        assert result == expected

    def test_custom_tags_configuration(self, custom_tags_processor):
        """Test using custom tags configuration."""
        original_block = "{custom}\n{special}"
        llm_block = "Some content"

        result = custom_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "Some content\n{custom}\n{special}"
        assert result == expected

    def test_empty_config(self):
        """Test with empty tags configuration."""
        processor = PreserveFStringTagsProcessor(config={"tags_to_preserve": []})
        original_block = "{preface}"
        llm_block = "Some content"

        result = processor.process(original_block=original_block, llm_block=llm_block)

        # Should return LLM block unchanged since no tags to preserve
        assert result == llm_block

    def test_none_config(self):
        """Test with None config."""
        processor = PreserveFStringTagsProcessor(config=None)
        original_block = "{preface}"
        llm_block = "Some content"

        result = processor.process(original_block=original_block, llm_block=llm_block)

        expected = "Some content\n{preface}"
        assert result == expected

    def test_multiple_occurrences_of_same_tag(self, preserve_fstring_tags_processor):
        """Test handling multiple occurrences of the same tag."""
        original_block = "{preface}\nSome content\n{preface}"
        llm_block = "Some content"

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "Some content\n{preface}\n{preface}"
        assert result == expected

    def test_tags_with_whitespace_variations(self, preserve_fstring_tags_processor):
        """Test handling tags with different whitespace patterns."""
        original_block = "  {preface}  "
        llm_block = "Some content"

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "Some content\n  {preface}  "
        assert result == expected

    def test_tags_in_complex_markdown(self, preserve_fstring_tags_processor):
        """Test handling tags in complex markdown content."""
        original_block = """# Header

{preface}

This is a paragraph.

> **Quote:** This is a quote. **End quote.**

{license}

- List item
- Another item"""

        llm_block = """# Header

This is a paragraph.

> **Quote:** This is a quote. **End quote.**

- List item
- Another item"""

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = """# Header

{preface}

This is a paragraph.

> **Quote:** This is a quote. **End quote.**

{license}

- List item
- Another item"""
        assert result == expected

    def test_similarity_threshold(self, preserve_fstring_tags_processor):
        """Test the similarity threshold for line matching."""
        original_block = "Some content\n{preface}\nMore content"
        llm_block = "Some content\nMore content"

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "Some content\n{preface}\nMore content"
        assert result == expected

    def test_dissimilar_lines(self, preserve_fstring_tags_processor):
        """Test handling of completely dissimilar lines."""
        original_block = "Original content\n{preface}"
        llm_block = "Completely different content with no similarity."

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        expected = "Completely different content with no similarity.\n{preface}"
        assert result == expected

    def test_empty_blocks(self, preserve_fstring_tags_processor):
        """Test handling of empty blocks."""
        original_block = ""
        llm_block = ""

        result = preserve_fstring_tags_processor.process(original_block=original_block, llm_block=llm_block)

        assert result == ""

    def test_processor_string_representation(self, preserve_fstring_tags_processor):
        """Test string representation of the processor."""
        assert "PreserveFStringTagsProcessor" in str(preserve_fstring_tags_processor)
        assert "preserve_fstring_tags" in str(preserve_fstring_tags_processor)


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
        original_block = "Original content"
        llm_block = "LLM content"

        result = post_processor_chain.process(original_block=original_block, llm_block=llm_block)

        assert result == llm_block

    def test_single_processor(self, post_processor_chain):
        """Test chain with single processor."""
        processor = NoNewHeadersPostProcessor()
        post_processor_chain.add_processor(processor=processor)

        original_block = "Content"
        llm_block = "# New Header\n\nContent"

        result = post_processor_chain.process(original_block=original_block, llm_block=llm_block)

        assert "# New Header" not in result
        assert "Content" in result

    def test_multiple_processors(self, post_processor_chain):
        """Test chain with multiple processors."""
        post_processor_chain.add_processor(processor=RemoveXmlTagsProcessor())
        post_processor_chain.add_processor(processor=RemoveTrailingWhitespaceProcessor())

        original_block = "Content"
        llm_block = "<p>Content</p>  \n\n"

        result = post_processor_chain.process(original_block=original_block, llm_block=llm_block)

        assert "<p>" not in result
        assert "</p>" not in result
        assert result.endswith("Content\n\n")

    def test_chain_initialization_with_processors(self, sample_processors):
        """Test chain initialization with processors."""
        chain = PostProcessorChain(processors=sample_processors)
        assert len(chain) == 2

    def test_add_processor(self, post_processor_chain):
        """Test adding processors to chain."""
        assert len(post_processor_chain) == 0

        processor = NoNewHeadersPostProcessor()
        post_processor_chain.add_processor(processor=processor)
        assert len(post_processor_chain) == 1

    def test_chain_processing_order(self, post_processor_chain):
        """Test that processors are applied in correct order."""

        # Create a custom processor that adds markers to track order
        class OrderTracker(PostProcessor):
            def __init__(self, name):
                super().__init__(name=name)
                self.order = []

            def process(self, original_block, llm_block, **kwargs):
                self.order.append(self.name)
                return llm_block

        tracker1 = OrderTracker(name="first")
        tracker2 = OrderTracker(name="second")

        post_processor_chain.add_processor(processor=tracker1)
        post_processor_chain.add_processor(processor=tracker2)

        original_block = "Original"
        llm_block = "LLM"

        result = post_processor_chain.process(original_block=original_block, llm_block=llm_block)

        assert result == "LLM"
        assert tracker1.order == ["first"]
        assert tracker2.order == ["second"]

    def test_chain_with_error_handling(self, post_processor_chain):
        """Test chain continues processing when a processor fails."""

        class ErrorProcessor(PostProcessor):
            def __init__(self, name):
                super().__init__(name=name)

            def process(self, original_block, llm_block, **kwargs):
                raise Exception(f"Error in {self.name}")

        class SafeProcessor(PostProcessor):
            def __init__(self, name):
                super().__init__(name=name)

            def process(self, original_block, llm_block, **kwargs):
                return llm_block

        error_processor = ErrorProcessor(name="error")
        safe_processor = SafeProcessor(name="safe")

        post_processor_chain.add_processor(processor=error_processor)
        post_processor_chain.add_processor(processor=safe_processor)

        original_block = "Original"
        llm_block = "LLM"

        # Should not raise an exception, should return original llm_block
        result = post_processor_chain.process(original_block=original_block, llm_block=llm_block)

        assert result == llm_block

    def test_chain_string_representation(self, post_processor_chain):
        """Test string representation of chain."""
        post_processor_chain.add_processor(processor=RemoveXmlTagsProcessor())
        post_processor_chain.add_processor(processor=RemoveTrailingWhitespaceProcessor())

        expected = "PostProcessorChain(['remove_xml_tags', 'remove_trailing_whitespace'])"
        assert str(post_processor_chain) == expected

    def test_chain_length(self, post_processor_chain):
        """Test chain length property."""
        assert len(post_processor_chain) == 0
        post_processor_chain.add_processor(processor=RemoveXmlTagsProcessor())
        assert len(post_processor_chain) == 1
        post_processor_chain.add_processor(processor=RemoveTrailingWhitespaceProcessor())
        assert len(post_processor_chain) == 2

    def test_chain_with_kwargs(self, post_processor_chain):
        """Test chain passes kwargs to processors."""

        class KwargsTracker(PostProcessor):
            def __init__(self, name):
                super().__init__(name=name)
                self.received_kwargs = {}

            def process(self, original_block, llm_block, **kwargs):
                self.received_kwargs = kwargs
                return llm_block

        tracker = KwargsTracker(name="tracker")
        post_processor_chain.add_processor(processor=tracker)

        original_block = "Original"
        llm_block = "LLM"

        result = post_processor_chain.process(original_block=original_block, llm_block=llm_block, test_param="custom")

        assert result == llm_block
        assert tracker.received_kwargs["test_param"] == "custom"

    def test_chain_with_empty_llm_block(self, post_processor_chain):
        """Test chain with empty LLM block."""
        post_processor_chain.add_processor(processor=RemoveXmlTagsProcessor())

        original_block = "Original content."
        llm_block = ""
        result = post_processor_chain.process(original_block=original_block, llm_block=llm_block)
        assert result == ""

    def test_chain_with_empty_original_block(self, post_processor_chain):
        """Test chain with empty original block."""
        post_processor_chain.add_processor(processor=RemoveXmlTagsProcessor())

        original_block = ""
        llm_block = "<p>Some content.</p>"
        expected_output = "Some content."
        result = post_processor_chain.process(original_block=original_block, llm_block=llm_block)
        assert result == expected_output

    def test_chain_with_complex_processors(self, post_processor_chain):
        """Test chain with complex processors that modify content significantly."""
        post_processor_chain.add_processor(processor=RemoveXmlTagsProcessor())
        post_processor_chain.add_processor(processor=RemoveTrailingWhitespaceProcessor())
        post_processor_chain.add_processor(processor=NoNewHeadersPostProcessor())

        original_block = "Regular text.\nSubsection text."
        llm_block = "<p>Regular text.</p>\n# Subsection text.\n## New header.\n<br>\n  Trailing spaces  "

        expected_output = "Regular text.\nSubsection text.\n<br>\n  Trailing spaces"
        result = post_processor_chain.process(original_block=original_block, llm_block=llm_block)
        assert result == expected_output
