import pytest

from src.post_processors import (
    EnsureBlankLineProcessor,
    OrderQuoteAnnotationProcessor,
    RemoveBlankLinesInListProcessor,
    RemoveTrailingWhitespaceProcessor,
    RemoveXmlTagsProcessor,
)

# ============================================================================
# EnsureBlankLineProcessor Tests
# ============================================================================


@pytest.fixture
def ensure_blank_line_processor():
    return EnsureBlankLineProcessor()


class TestEnsureBlankLineProcessor:
    """Test suite for EnsureBlankLineProcessor."""

    def test_ensure_blank_line_between_elements(self, ensure_blank_line_processor):
        """Test basic blank line insertion between paragraphs."""
        llm_block = "Paragraph 1.\nParagraph 2."
        expected_output = "Paragraph 1.\n\nParagraph 2."
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_no_blank_line_for_list_items(self, ensure_blank_line_processor):
        """Test that list items don't get blank lines between them."""
        llm_block = "* Item 1\n* Item 2"
        expected_output = "* Item 1\n* Item 2"
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_no_blank_line_for_multiline_quote(self, ensure_blank_line_processor):
        """Test that multiline quotes don't get blank lines between lines."""
        llm_block = "> Quote line 1\n> Quote line 2"
        expected_output = "> Quote line 1\n> Quote line 2"
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_blank_line_between_special_blocks(self, ensure_blank_line_processor):
        """Test blank lines between Quote and Annotation blocks."""
        llm_block = "> **Quote:** A quote. **End quote.**\n> **Annotation:** An annotation. **End annotation.**"
        expected_output = "> **Quote:** A quote. **End quote.**\n\n> **Annotation:** An annotation. **End annotation.**"
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_quote_block_surrounded_by_blank_lines(self, ensure_blank_line_processor):
        """Test that Quote blocks are properly surrounded by blank lines."""
        llm_block = "Some text.\n> **Quote:** This is a quote. **End quote.**\nMore text."
        expected_output = "Some text.\n\n> **Quote:** This is a quote. **End quote.**\n\nMore text."
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_annotation_block_surrounded_by_blank_lines(self, ensure_blank_line_processor):
        """Test that Annotation blocks are properly surrounded by blank lines."""
        llm_block = "Some text.\n> **Annotation:** This is an annotation. **End annotation.**\nMore text."
        expected_output = "Some text.\n\n> **Annotation:** This is an annotation. **End annotation.**\n\nMore text."
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_multiple_quote_blocks_with_blank_lines(self, ensure_blank_line_processor):
        """Test multiple Quote/Annotation blocks with proper blank line separation."""
        llm_block = (
            "Introduction.\n"
            "> **Quote:** First quote. **End quote.**\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "Conclusion."
        )
        expected_output = (
            "Introduction.\n\n"
            "> **Quote:** First quote. **End quote.**\n\n"
            "> **Annotation:** First annotation. **End annotation.**\n\n"
            "> **Quote:** Second quote. **End quote.**\n\n"
            "Conclusion."
        )
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_quote_block_with_multiple_lines(self, ensure_blank_line_processor):
        """Test that Quote blocks are single-line and don't have internal blank lines."""
        llm_block = "Text before.\n> **Quote:** This is a single-line quote block. **End quote.**\nText after."
        expected_output = (
            "Text before.\n\n> **Quote:** This is a single-line quote block. **End quote.**\n\nText after."
        )
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_annotation_block_with_multiple_lines(self, ensure_blank_line_processor):
        """Test that Annotation blocks are single-line and don't have internal blank lines."""
        llm_block = (
            "Text before.\n> **Annotation:** This is a single-line annotation block. **End annotation.**\nText after."
        )
        expected_output = (
            "Text before.\n\n"
            "> **Annotation:** This is a single-line annotation block. **End annotation.**\n\n"
            "Text after."
        )
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_regular_quotes_not_surrounded(self, ensure_blank_line_processor):
        """Test that regular quotes (not Quote/Annotation blocks) are not surrounded."""
        llm_block = "Some text.\n> This is a regular quote.\n> Not a special block.\nMore text."
        expected_output = "Some text.\n\n> This is a regular quote.\n> Not a special block.\n\nMore text."
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_mixed_content_with_quotes_and_lists(self, ensure_blank_line_processor):
        """Test mixed content with quotes, lists, and regular text."""
        llm_block = (
            "Introduction.\n"
            "* List item 1\n"
            "* List item 2\n"
            "> **Quote:** A quote. **End quote.**\n"
            "Regular paragraph.\n"
            "> **Annotation:** An annotation. **End annotation.**\n"
            "Conclusion."
        )
        expected_output = (
            "Introduction.\n\n"
            "* List item 1\n"
            "* List item 2\n\n"
            "> **Quote:** A quote. **End quote.**\n\n"
            "Regular paragraph.\n\n"
            "> **Annotation:** An annotation. **End annotation.**\n\n"
            "Conclusion."
        )
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_empty_lines_preserved(self, ensure_blank_line_processor):
        """Test that existing empty lines are preserved."""
        llm_block = "Line 1.\n\nLine 2.\n\n\nLine 3."
        expected_output = "Line 1.\n\nLine 2.\n\n\nLine 3."
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_quote_block_at_end(self, ensure_blank_line_processor):
        """Test Quote block at the end of content."""
        llm_block = "Some text.\n> **Quote:** Final quote. **End quote.**"
        expected_output = "Some text.\n\n> **Quote:** Final quote. **End quote.**"
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_quote_block_at_beginning(self, ensure_blank_line_processor):
        """Test Quote block at the beginning of content."""
        llm_block = "> **Quote:** Opening quote. **End quote.**\nSome text."
        expected_output = "> **Quote:** Opening quote. **End quote.**\n\nSome text."
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_multiline_quotes_still_work(self, ensure_blank_line_processor):
        """Test that regular multiline quotes (not Quote/Annotation blocks) still work."""
        llm_block = (
            "Some text.\n"
            "> This is a regular multiline quote.\n"
            "> It continues on multiple lines.\n"
            "> It should not have blank lines between lines.\n"
            "More text."
        )
        expected_output = (
            "Some text.\n\n"
            "> This is a regular multiline quote.\n"
            "> It continues on multiple lines.\n"
            "> It should not have blank lines between lines.\n\n"
            "More text."
        )
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_mixed_list_types(self, ensure_blank_line_processor):
        """Test mixed list types (asterisk and dash)."""
        llm_block = "* Item 1\n- Item 2\n* Item 3"
        expected_output = "* Item 1\n- Item 2\n* Item 3"
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_list_with_nested_content(self, ensure_blank_line_processor):
        """Test lists with nested content.

        Note: Nested content (indented text that's not a list item) requires
        blank line separation because it's not part of the list item exception.
        The processor only avoids blank lines between consecutive list items
        (lines starting with '* ' or '- '), not between list items and their
        nested content. This improves readability by clearly separating the
        list structure from its nested content.
        """
        llm_block = "* Item 1\n  Nested content\n* Item 2"
        expected_output = "* Item 1\n\n  Nested content\n\n* Item 2"
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_quote_block_with_whitespace(self, ensure_blank_line_processor):
        """Test Quote blocks with leading/trailing whitespace."""
        llm_block = "Text.\n> **Quote:**  Quote with spaces  . **End quote.**\nMore text."
        expected_output = "Text.\n\n> **Quote:**  Quote with spaces  . **End quote.**\n\nMore text."
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_empty_input(self, ensure_blank_line_processor):
        """Test with empty input."""
        llm_block = ""
        expected_output = ""
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_single_line_input(self, ensure_blank_line_processor):
        """Test with single line input."""
        llm_block = "Single line."
        expected_output = "Single line."
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_consecutive_blank_lines(self, ensure_blank_line_processor):
        """Test that consecutive blank lines are preserved."""
        llm_block = "Line 1.\n\n\nLine 2."
        expected_output = "Line 1.\n\n\nLine 2."
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_quote_block_with_trailing_whitespace(self, ensure_blank_line_processor):
        """Test Quote blocks with trailing whitespace are handled correctly."""
        llm_block = "Text.\n> **Quote:** Quote with trailing spaces   . **End quote.**\nMore text."
        expected_output = "Text.\n\n> **Quote:** Quote with trailing spaces   . **End quote.**\n\nMore text."
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_annotation_block_with_leading_whitespace(self, ensure_blank_line_processor):
        """Test Annotation blocks with leading whitespace are handled correctly."""
        llm_block = "Text.\n>   **Annotation:** Annotation with leading spaces. **End annotation.**\nMore text."
        expected_output = (
            "Text.\n\n>   **Annotation:** Annotation with leading spaces. **End annotation.**\n\nMore text."
        )
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_mixed_whitespace_in_quotes(self, ensure_blank_line_processor):
        """Test that mixed whitespace in regular quotes is handled correctly."""
        llm_block = "Text.\n>   Quote with mixed whitespace  .\n>   Another line with spaces  .\nMore text."
        expected_output = "Text.\n\n>   Quote with mixed whitespace  .\n>   Another line with spaces  .\n\nMore text."
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_unicode_and_emoji_handling(self, ensure_blank_line_processor):
        """Test handling of Unicode characters, emojis, and non-Latin scripts."""
        llm_block = (
            "Introduction with café ☕\n"
            '> **Quote:** 中文引用："你好世界" **End quote.**\n'
            "Arabic text: مرحبا بالعالم\n"
            "> **Annotation:** Русский: Привет мир! 🌍 **End annotation.**\n"
            "Japanese: こんにちは 📚"
        )
        expected_output = (
            "Introduction with café ☕\n\n"
            '> **Quote:** 中文引用："你好世界" **End quote.**\n\n'
            "Arabic text: مرحبا بالعالم\n\n"
            "> **Annotation:** Русский: Привет мир! 🌍 **End annotation.**\n\n"
            "Japanese: こんにちは 📚"
        )
        result = ensure_blank_line_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output


# ============================================================================
# RemoveBlankLinesInListProcessor Tests
# ============================================================================


@pytest.fixture
def remove_blank_lines_in_list_processor():
    return RemoveBlankLinesInListProcessor()


class TestRemoveBlankLinesInListProcessor:
    """Test suite for RemoveBlankLinesInListProcessor."""

    def test_remove_blank_line_between_list_items(self, remove_blank_lines_in_list_processor):
        """Test basic blank line removal between list items."""
        llm_block = "* Item 1\n\n* Item 2"
        expected_output = "* Item 1\n* Item 2"
        result = remove_blank_lines_in_list_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_remove_multiple_blank_lines_between_list_items(self, remove_blank_lines_in_list_processor):
        """Test removal of multiple blank lines between list items."""
        llm_block = "* Item 1\n\n\n* Item 2"
        expected_output = "* Item 1\n* Item 2"
        result = remove_blank_lines_in_list_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_no_blank_lines_to_remove(self, remove_blank_lines_in_list_processor):
        """Test list with no blank lines."""
        llm_block = "* Item 1\n* Item 2\n* Item 3"
        expected_output = "* Item 1\n* Item 2\n* Item 3"
        result = remove_blank_lines_in_list_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_blank_lines_outside_list_preserved(self, remove_blank_lines_in_list_processor):
        """Test that blank lines outside the list are preserved."""
        llm_block = "Some text.\n\n* Item 1\n\n* Item 2\n\nMore text."
        expected_output = "Some text.\n\n* Item 1\n* Item 2\n\nMore text."
        result = remove_blank_lines_in_list_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_mixed_list_types_with_blank_lines(self, remove_blank_lines_in_list_processor):
        """Test mixed list types with blank lines."""
        llm_block = "* Item 1\n\n- Item 2\n\n+ Item 3"
        expected_output = "* Item 1\n- Item 2\n+ Item 3"
        result = remove_blank_lines_in_list_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_ordered_list_with_blank_lines(self, remove_blank_lines_in_list_processor):
        """Test ordered list with blank lines."""
        llm_block = "1. Item 1\n\n2. Item 2\n\n3. Item 3"
        expected_output = "1. Item 1\n2. Item 2\n3. Item 3"
        result = remove_blank_lines_in_list_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_list_with_nested_content_and_blank_lines(self, remove_blank_lines_in_list_processor):
        """Test list with nested content and blank lines."""
        llm_block = "* Item 1\n\n  Nested content\n\n* Item 2"
        expected_output = "* Item 1\n  Nested content\n* Item 2"
        result = remove_blank_lines_in_list_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_empty_input(self, remove_blank_lines_in_list_processor):
        """Test with empty input."""
        llm_block = ""
        expected_output = ""
        result = remove_blank_lines_in_list_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_no_list_input(self, remove_blank_lines_in_list_processor):
        """Test input with no list."""
        llm_block = "Some text.\n\nMore text."
        expected_output = "Some text.\n\nMore text."
        result = remove_blank_lines_in_list_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_unicode_list_items(self, remove_blank_lines_in_list_processor):
        """Test list processing with Unicode characters and emojis."""
        llm_block = (
            "* 咖啡 ☕ - Chinese coffee\n\n"
            "* العربية - Arabic text\n\n"
            "* Русский язык 🇷🇺\n\n"
            "* 日本語 📖 with nested\n\n"
            "  Indented content こんにちは\n\n"
            "* Émojis and accénts"
        )
        expected_output = (
            "* 咖啡 ☕ - Chinese coffee\n"
            "* العربية - Arabic text\n"
            "* Русский язык 🇷🇺\n"
            "* 日本語 📖 with nested\n"
            "  Indented content こんにちは\n"
            "* Émojis and accénts"
        )
        result = remove_blank_lines_in_list_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output


# ============================================================================
# RemoveXmlTagsProcessor Tests
# ============================================================================


@pytest.fixture
def remove_xml_tags_processor():
    return RemoveXmlTagsProcessor()


class TestRemoveXmlTagsProcessor:
    """Test suite for RemoveXmlTagsProcessor."""

    def test_remove_xml_tags(self, remove_xml_tags_processor):
        """Test basic XML tag removal."""
        llm_block = "<p>Some text.</p>\n<br>\n<item>Another item</item>"
        expected_output = "Some text.\n<br>\nAnother item"
        result = remove_xml_tags_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_preserve_br_tags(self, remove_xml_tags_processor):
        """Test that <br> tags are preserved."""
        llm_block = "<p>Text</p>\n<br>\n<br/>\n<strong>Bold</strong>"
        expected_output = "Text\n<br>\n<br/>\nBold"
        result = remove_xml_tags_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_remove_self_closing_tags(self, remove_xml_tags_processor):
        """Test removal of self-closing tags."""
        llm_block = "<img src='test.jpg'/>\n<hr/>\n<br>\nText"
        expected_output = "\n\n<br>\nText"
        result = remove_xml_tags_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_remove_nested_tags(self, remove_xml_tags_processor):
        """Test removal of nested XML tags."""
        llm_block = "<div><p><strong>Bold text</strong></p></div>"
        expected_output = "Bold text"
        result = remove_xml_tags_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_remove_tags_with_attributes(self, remove_xml_tags_processor):
        """Test removal of tags with attributes."""
        llm_block = '<p class="test" id="main">Content</p>'
        expected_output = "Content"
        result = remove_xml_tags_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_remove_tags_with_special_characters(self, remove_xml_tags_processor):
        """Test removal of tags with special characters in attributes."""
        llm_block = '<a href="test.html" onclick="alert(\'test\')">Link</a>'
        expected_output = "Link"
        result = remove_xml_tags_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_no_tags_to_remove(self, remove_xml_tags_processor):
        """Test text with no XML tags."""
        llm_block = "Plain text without any tags."
        expected_output = "Plain text without any tags."
        result = remove_xml_tags_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_empty_input(self, remove_xml_tags_processor):
        """Test with empty input."""
        llm_block = ""
        expected_output = ""
        result = remove_xml_tags_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_only_tags(self, remove_xml_tags_processor):
        """Test input containing only XML tags."""
        llm_block = "<p></p>\n<div></div>\n<br>"
        expected_output = "\n\n<br>"
        result = remove_xml_tags_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_malformed_tags(self, remove_xml_tags_processor):
        """Test handling of malformed XML tags."""
        llm_block = "<p>Text</p>\n<unclosed>\n<br>\n<malformed"
        expected_output = "Text\n\n<br>\n<malformed"
        result = remove_xml_tags_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_unicode_content_in_xml_tags(self, remove_xml_tags_processor):
        """Test XML tag removal with Unicode content inside tags."""
        llm_block = (
            '<p class="chinese">中文内容</p>\n'
            '<div lang="ar">النص العربي</div>\n'
            "<span>Русский текст с émojis 🌟</span>\n"
            "<br>\n"
            "<strong>日本語: こんにちは</strong>"
        )
        expected_output = "中文内容\nالنص العربي\nРусский текст с émojis 🌟\n<br>\n日本語: こんにちは"
        result = remove_xml_tags_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output


# ============================================================================
# RemoveTrailingWhitespaceProcessor Tests
# ============================================================================


@pytest.fixture
def remove_trailing_whitespace_processor():
    return RemoveTrailingWhitespaceProcessor()


class TestRemoveTrailingWhitespaceProcessor:
    """Test suite for RemoveTrailingWhitespaceProcessor."""

    def test_remove_trailing_whitespace(self, remove_trailing_whitespace_processor):
        """Test basic trailing whitespace removal."""
        llm_block = "Line 1  \nLine 2\t\nLine 3"
        expected_output = "Line 1\nLine 2\nLine 3"
        result = remove_trailing_whitespace_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_remove_mixed_whitespace(self, remove_trailing_whitespace_processor):
        """Test removal of mixed whitespace characters."""
        llm_block = "Line 1 \t \nLine 2  \t  \nLine 3"
        expected_output = "Line 1\nLine 2\nLine 3"
        result = remove_trailing_whitespace_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_preserve_leading_whitespace(self, remove_trailing_whitespace_processor):
        """Test that leading whitespace is preserved."""
        llm_block = "  Line 1  \n\tLine 2\t\n   Line 3   "
        expected_output = "  Line 1\n\tLine 2\n   Line 3"
        result = remove_trailing_whitespace_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_preserve_leading_and_remove_trailing(self, remove_trailing_whitespace_processor):
        """Test that leading whitespace is preserved and trailing whitespace is removed."""
        llm_block = "  Line 1  \n\tLine 2\t\n   Line 3   \n"
        expected_output = "  Line 1\n\tLine 2\n   Line 3\n"
        result = remove_trailing_whitespace_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_empty_lines(self, remove_trailing_whitespace_processor):
        """Test handling of empty lines."""
        llm_block = "Line 1\n  \n\t\nLine 2"
        expected_output = "Line 1\n\n\nLine 2"
        result = remove_trailing_whitespace_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_only_whitespace_lines(self, remove_trailing_whitespace_processor):
        """Test lines containing only whitespace."""
        llm_block = "  \n\t\n   \n"
        expected_output = "\n\n\n"
        result = remove_trailing_whitespace_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_no_trailing_whitespace(self, remove_trailing_whitespace_processor):
        """Test text with no trailing whitespace."""
        llm_block = "Line 1\nLine 2\nLine 3"
        expected_output = "Line 1\nLine 2\nLine 3"
        result = remove_trailing_whitespace_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_empty_input(self, remove_trailing_whitespace_processor):
        """Test with empty input."""
        llm_block = ""
        expected_output = ""
        result = remove_trailing_whitespace_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output


# ============================================================================
# OrderQuoteAnnotationProcessor Tests
# ============================================================================


@pytest.fixture
def order_quote_annotation_processor():
    return OrderQuoteAnnotationProcessor()


class TestOrderQuoteAnnotationProcessor:
    """Test suite for OrderQuoteAnnotationProcessor."""

    def test_simple_reorder_annotation_before_quote(self, order_quote_annotation_processor):
        """Test reordering when annotation comes before quote."""
        llm_block = (
            "Some text.\n"
            "> **Annotation:** This is an annotation. **End annotation.**\n"
            "> **Quote:** This is a quote. **End quote.**\n"
            "More text."
        )
        expected_output = (
            "Some text.\n"
            "> **Quote:** This is a quote. **End quote.**\n"
            "> **Annotation:** This is an annotation. **End annotation.**\n"
            "More text."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_simple_reorder_annotation_before_quote_with_newline(self, order_quote_annotation_processor):
        """Test reordering when annotation comes before quote with blank line."""
        llm_block = (
            "Some text.\n"
            "> **Annotation:** This is an annotation. **End annotation.**\n\n"
            "> **Quote:** This is a quote. **End quote.**\n"
            "More text."
        )
        expected_output = (
            "Some text.\n"
            "> **Quote:** This is a quote. **End quote.**\n"
            "> **Annotation:** This is an annotation. **End annotation.**\n"
            "More text."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_multiple_quotes_and_annotations(self, order_quote_annotation_processor):
        """Test reordering multiple quotes and annotations in mixed order."""
        llm_block = (
            "Some text.\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "> **Quote:** First quote. **End quote.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "More text."
        )
        expected_output = (
            "Some text.\n"
            "> **Quote:** First quote. **End quote.**\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "More text."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_separate_blocks_not_affected(self, order_quote_annotation_processor):
        """Test that separate blocks (separated by blank lines) are not affected."""
        llm_block = (
            "First paragraph.\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "> **Quote:** First quote. **End quote.**\n"
            "\n"
            "Second paragraph.\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "Third paragraph."
        )
        expected_output = (
            "First paragraph.\n"
            "> **Quote:** First quote. **End quote.**\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "\n"
            "Second paragraph.\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "Third paragraph."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_quotes_only_no_change(self, order_quote_annotation_processor):
        """Test that blocks with only quotes are not changed."""
        llm_block = (
            "Some text.\n"
            "> **Quote:** First quote. **End quote.**\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "More text."
        )
        expected_output = (
            "Some text.\n"
            "> **Quote:** First quote. **End quote.**\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "More text."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_annotations_only_no_change(self, order_quote_annotation_processor):
        """Test that blocks with only annotations are not changed."""
        llm_block = (
            "Some text.\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "More text."
        )
        expected_output = (
            "Some text.\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "More text."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_mixed_content_with_regular_quotes(self, order_quote_annotation_processor):
        """Test that regular quotes (not Quote/Annotation blocks) break uninterrupted blocks."""
        llm_block = (
            "Some text.\n"
            "> **Annotation:** An annotation. **End annotation.**\n"
            "> This is a regular quote.\n"
            "> **Quote:** A quote block. **End quote.**\n"
            "More text."
        )
        expected_output = (
            "Some text.\n"
            "> **Annotation:** An annotation. **End annotation.**\n"
            "> This is a regular quote.\n"
            "> **Quote:** A quote block. **End quote.**\n"
            "More text."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_empty_input(self, order_quote_annotation_processor):
        """Test with empty input."""
        llm_block = ""
        expected_output = ""
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_no_quotes_or_annotations(self, order_quote_annotation_processor):
        """Test with no Quote or Annotation blocks."""
        llm_block = "First paragraph.\nSecond paragraph.\n> Regular quote.\nThird paragraph."
        expected_output = "First paragraph.\nSecond paragraph.\n> Regular quote.\nThird paragraph."
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_complex_mixed_scenario(self, order_quote_annotation_processor):
        """Test a complex scenario with multiple blocks and mixed content."""
        llm_block = (
            "Introduction.\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "> **Quote:** First quote. **End quote.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "\n"
            "Middle section.\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "> **Annotation:** Third annotation. **End annotation.**\n"
            "> **Quote:** Third quote. **End quote.**\n"
            "> **Annotation:** Fourth annotation. **End annotation.**\n"
            "\n"
            "Conclusion.\n"
            "> **Annotation:** Final annotation. **End annotation.**\n"
            "> **Quote:** Final quote. **End quote.**\n"
            "End."
        )
        expected_output = (
            "Introduction.\n"
            "> **Quote:** First quote. **End quote.**\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "\n"
            "Middle section.\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "> **Quote:** Third quote. **End quote.**\n"
            "> **Annotation:** Third annotation. **End annotation.**\n"
            "> **Annotation:** Fourth annotation. **End annotation.**\n"
            "\n"
            "Conclusion.\n"
            "> **Quote:** Final quote. **End quote.**\n"
            "> **Annotation:** Final annotation. **End annotation.**\n"
            "End."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_quote_annotation_with_whitespace(self, order_quote_annotation_processor):
        """Test Quote/Annotation blocks with leading/trailing whitespace."""
        llm_block = (
            "Text.\n"
            "> **Annotation:**  Annotation with spaces  . **End annotation.**\n"
            "> **Quote:**  Quote with spaces  . **End quote.**\n"
            "More text."
        )
        expected_output = (
            "Text.\n"
            "> **Quote:**  Quote with spaces  . **End quote.**\n"
            "> **Annotation:**  Annotation with spaces  . **End annotation.**\n"
            "More text."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_multiple_consecutive_blank_lines(self, order_quote_annotation_processor):
        """Test handling of multiple consecutive blank lines."""
        llm_block = (
            "Text.\n"
            "> **Annotation:** An annotation. **End annotation.**\n"
            "\n\n"
            "> **Quote:** A quote. **End quote.**\n"
            "More text."
        )
        expected_output = (
            "Text.\n"
            "> **Quote:** A quote. **End quote.**\n"
            "> **Annotation:** An annotation. **End annotation.**\n"
            "More text."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_quote_annotation_at_beginning(self, order_quote_annotation_processor):
        """Test Quote/Annotation blocks at the beginning of content."""
        llm_block = (
            "> **Annotation:** First annotation. **End annotation.**\n"
            "> **Quote:** First quote. **End quote.**\n"
            "Regular text."
        )
        expected_output = (
            "> **Quote:** First quote. **End quote.**\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "Regular text."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_quote_annotation_at_end(self, order_quote_annotation_processor):
        """Test Quote/Annotation blocks at the end of content."""
        llm_block = (
            "Regular text.\n"
            "> **Annotation:** Final annotation. **End annotation.**\n"
            "> **Quote:** Final quote. **End quote.**"
        )
        expected_output = (
            "Regular text.\n"
            "> **Quote:** Final quote. **End quote.**\n"
            "> **Annotation:** Final annotation. **End annotation.**"
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_multiple_consecutive_blank_lines_between_blocks(self, order_quote_annotation_processor):
        """Test handling of multiple consecutive blank lines between quote/annotation blocks."""
        llm_block = (
            "First paragraph.\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "> **Quote:** First quote. **End quote.**\n"
            "\n\n\n"
            "Second paragraph.\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "\n\n"
            "Third paragraph.\n"
            "> **Annotation:** Third annotation. **End annotation.**\n"
            "> **Quote:** Third quote. **End quote.**\n"
            "Final text."
        )
        expected_output = (
            "First paragraph.\n"
            "> **Quote:** First quote. **End quote.**\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "\n\n\n"
            "Second paragraph.\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "\n\n"
            "Third paragraph.\n"
            "> **Quote:** Third quote. **End quote.**\n"
            "> **Annotation:** Third annotation. **End annotation.**\n"
            "Final text."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_whitespace_only_lines_between_blocks(self, order_quote_annotation_processor):
        """Test handling of lines with only spaces or tabs between quote/annotation blocks."""
        llm_block = (
            "First paragraph.\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "> **Quote:** First quote. **End quote.**\n"
            "   \n"
            "Second paragraph.\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "\t\n"
            "Third paragraph.\n"
            "> **Annotation:** Third annotation. **End annotation.**\n"
            "> **Quote:** Third quote. **End quote.**\n"
            "Final text."
        )
        expected_output = (
            "First paragraph.\n"
            "> **Quote:** First quote. **End quote.**\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "   \n"
            "Second paragraph.\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "\t\n"
            "Third paragraph.\n"
            "> **Quote:** Third quote. **End quote.**\n"
            "> **Annotation:** Third annotation. **End annotation.**\n"
            "Final text."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_quote_annotation_blocks_at_start_with_blank_lines(self, order_quote_annotation_processor):
        """Test Quote/Annotation blocks at the very start with blank lines after."""
        llm_block = (
            "> **Annotation:** Starting annotation. **End annotation.**\n"
            "> **Quote:** Starting quote. **End quote.**\n"
            "\n\n"
            "Regular text follows.\n"
            "> **Quote:** Middle quote. **End quote.**\n"
            "> **Annotation:** Middle annotation. **End annotation.**\n"
            "More text."
        )
        expected_output = (
            "> **Quote:** Starting quote. **End quote.**\n"
            "> **Annotation:** Starting annotation. **End annotation.**\n"
            "\n\n"
            "Regular text follows.\n"
            "> **Quote:** Middle quote. **End quote.**\n"
            "> **Annotation:** Middle annotation. **End annotation.**\n"
            "More text."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_quote_annotation_blocks_at_end_with_blank_lines(self, order_quote_annotation_processor):
        """Test Quote/Annotation blocks at the very end with blank lines before."""
        llm_block = (
            "Regular text starts.\n"
            "> **Quote:** Middle quote. **End quote.**\n"
            "> **Annotation:** Middle annotation. **End annotation.**\n"
            "More text.\n"
            "\n\n"
            "> **Annotation:** Final annotation. **End annotation.**\n"
            "> **Quote:** Final quote. **End quote.**"
        )
        expected_output = (
            "Regular text starts.\n"
            "> **Quote:** Middle quote. **End quote.**\n"
            "> **Annotation:** Middle annotation. **End annotation.**\n"
            "More text.\n"
            "\n\n"
            "> **Quote:** Final quote. **End quote.**\n"
            "> **Annotation:** Final annotation. **End annotation.**"
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_mixed_whitespace_and_blank_lines(self, order_quote_annotation_processor):
        """Test complex scenario with mixed whitespace-only lines and blank lines."""
        llm_block = (
            "> **Annotation:** Start annotation. **End annotation.**\n"
            "> **Quote:** Start quote. **End quote.**\n"
            "  \n"
            "First paragraph.\n"
            "> **Quote:** First quote. **End quote.**\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "\n"
            "\t\n"
            "Second paragraph.\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "   \n"
            "\n"
            "> **Quote:** End quote. **End quote.**\n"
            "> **Annotation:** End annotation. **End annotation.**"
        )
        expected_output = (
            "> **Quote:** Start quote. **End quote.**\n"
            "> **Annotation:** Start annotation. **End annotation.**\n"
            "  \n"
            "First paragraph.\n"
            "> **Quote:** First quote. **End quote.**\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "\n"
            "\t\n"
            "Second paragraph.\n"
            "> **Quote:** Second quote. **End quote.**\n"
            "> **Quote:** End quote. **End quote.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "> **Annotation:** End annotation. **End annotation.**"
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_lookahead_logic_with_multiple_blank_lines(self, order_quote_annotation_processor):
        """Test the lookahead logic when there are multiple blank lines between blocks."""
        llm_block = (
            "Text.\n"
            "> **Annotation:** An annotation. **End annotation.**\n"
            "\n\n\n\n"
            "> **Quote:** A quote. **End quote.**\n"
            "More text.\n"
            "> **Quote:** Another quote. **End quote.**\n"
            "> **Annotation:** Another annotation. **End annotation.**\n"
            "\n\n"
            "Final text."
        )
        expected_output = (
            "Text.\n"
            "> **Quote:** A quote. **End quote.**\n"
            "> **Annotation:** An annotation. **End annotation.**\n"
            "More text.\n"
            "> **Quote:** Another quote. **End quote.**\n"
            "> **Annotation:** Another annotation. **End annotation.**\n"
            "\n\n"
            "Final text."
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_flush_logic_at_boundaries(self, order_quote_annotation_processor):
        """Test flush logic when blocks are at content boundaries with complex spacing."""
        llm_block = (
            "> **Annotation:** First annotation. **End annotation.**\n"
            "> **Quote:** First quote. **End quote.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "\n"
            "Middle text.\n"
            "\n"
            "> **Quote:** Last quote. **End quote.**\n"
            "> **Annotation:** Last annotation. **End annotation.**"
        )
        expected_output = (
            "> **Quote:** First quote. **End quote.**\n"
            "> **Annotation:** First annotation. **End annotation.**\n"
            "> **Annotation:** Second annotation. **End annotation.**\n"
            "\n"
            "Middle text.\n"
            "\n"
            "> **Quote:** Last quote. **End quote.**\n"
            "> **Annotation:** Last annotation. **End annotation.**"
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_edge_case_only_whitespace_content(self, order_quote_annotation_processor):
        """Test edge case where content consists only of whitespace and quote/annotation blocks."""
        llm_block = (
            "   \n"
            "> **Annotation:** An annotation. **End annotation.**\n"
            "> **Quote:** A quote. **End quote.**\n"
            "\t\n"
            "> **Quote:** Another quote. **End quote.**\n"
            "> **Annotation:** Another annotation. **End annotation.**\n"
            "  "
        )
        expected_output = (
            "   \n"
            "> **Quote:** A quote. **End quote.**\n"
            "> **Quote:** Another quote. **End quote.**\n"
            "> **Annotation:** An annotation. **End annotation.**\n"
            "> **Annotation:** Another annotation. **End annotation.**\n"
            "  "
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output

    def test_unicode_content_handling(self, order_quote_annotation_processor):
        """Test handling of Unicode characters in various languages."""
        llm_block = (
            "Introduction with café and münü.\n"
            "> **Annotation:** 中文注释。 **End annotation.**\n"
            '> **Quote:** "العربية quote" **End quote.**\n'
            "Conclusion with émojis: 🎉📚\n"
            "> **Quote:** Russian: Привет мир **End quote.**\n"
            "> **Annotation:** Japanese: こんにちは世界 **End annotation.**"
        )
        # The processor treats each group separately (separated by non-quote content)
        expected_output = (
            "Introduction with café and münü.\n"
            '> **Quote:** "العربية quote" **End quote.**\n'
            "> **Annotation:** 中文注释。 **End annotation.**\n"
            "Conclusion with émojis: 🎉📚\n"
            "> **Quote:** Russian: Привет мир **End quote.**\n"
            "> **Annotation:** Japanese: こんにちは世界 **End annotation.**"
        )
        result = order_quote_annotation_processor.process(original_block="", llm_block=llm_block)
        assert result == expected_output
