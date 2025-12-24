"""Tests for BaseBookBuilder.clean_start_markers() method."""

import tempfile
from pathlib import Path

import pytest

from books.base_builder import BaseBookBuilder, BookConfig


# Mock builder class for testing
class MockBookBuilder(BaseBookBuilder):
    """Mock implementation of BaseBookBuilder for testing."""

    def get_source_files(self):
        """Return empty dict for testing."""
        return {}

    def get_original_file(self):
        """Return None for testing."""
        return None


@pytest.fixture
def temp_markdown_file():
    """Create a temporary markdown file for testing."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False, encoding="utf-8") as f:
        temp_path = Path(f.name)
        yield temp_path
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


@pytest.fixture
def builder():
    """Create a test builder instance."""
    config = BookConfig(
        name="test_book",
        version="1.0.0",
        title="Test Book",
        author="Test Author",
    )
    return MockBookBuilder(config=config)


class TestCleanStartMarkers:
    """Test suite for BaseBookBuilder.clean_start_markers()."""

    def test_replace_annotation_marker(self, builder, temp_markdown_file):
        """Test that **Annotation:** is replaced with **Note:**."""
        content = "Some text.\n\n> **Annotation:** This is an annotation. **End annotation.**\n\nMore text."
        temp_markdown_file.write_text(content, encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        expected = "Some text.\n\n> **Note:** This is an annotation. **End annotation.**\n\nMore text."
        assert result == expected

    def test_replace_annotated_introduction_marker(self, builder, temp_markdown_file):
        """Test that **Annotated introduction:** is replaced with **Introduction:**."""
        content = (
            "## Chapter 1\n\n"
            "> **Annotated introduction:**<br>\n"
            "> This is the introduction.\n"
            "> **End annotated introduction.**\n\n"
            "Chapter content."
        )
        temp_markdown_file.write_text(content, encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        expected = (
            "## Chapter 1\n\n"
            "> **Introduction:**<br>\n"
            "> This is the introduction.\n"
            "> **End annotated introduction.**\n\n"
            "Chapter content."
        )
        assert result == expected

    def test_replace_annotated_summary_marker(self, builder, temp_markdown_file):
        """Test that **Annotated summary:** is replaced with **Summary:**."""
        content = (
            "Chapter content.\n\n> **Annotated summary:**<br>\n> This is the summary.\n> **End annotated summary.**\n"
        )
        temp_markdown_file.write_text(content, encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        expected = "Chapter content.\n\n> **Summary:**<br>\n> This is the summary.\n> **End annotated summary.**\n"
        assert result == expected

    def test_keep_quote_marker_unchanged(self, builder, temp_markdown_file):
        """Test that **Quote:** markers are left unchanged."""
        content = 'Some text.\n\n> **Quote:** "This is a quote." **End quote.**\n\nMore text.'
        temp_markdown_file.write_text(content, encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        # Should be unchanged
        assert result == content

    def test_replace_multiple_markers_of_same_type(self, builder, temp_markdown_file):
        """Test replacing multiple instances of the same marker type."""
        content = (
            "Text.\n\n"
            "> **Annotation:** First annotation. **End annotation.**\n\n"
            "Middle text.\n\n"
            "> **Annotation:** Second annotation. **End annotation.**\n\n"
            "End text."
        )
        temp_markdown_file.write_text(content, encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        expected = (
            "Text.\n\n"
            "> **Note:** First annotation. **End annotation.**\n\n"
            "Middle text.\n\n"
            "> **Note:** Second annotation. **End annotation.**\n\n"
            "End text."
        )
        assert result == expected

    def test_replace_mixed_marker_types(self, builder, temp_markdown_file):
        """Test replacing multiple different marker types in one file."""
        content = (
            "# Chapter\n\n"
            "> **Annotated introduction:**<br>\n"
            "> Introduction text.\n"
            "> **End annotated introduction.**\n\n"
            "Main content.\n\n"
            "> **Annotation:** A note about something. **End annotation.**\n\n"
            '> **Quote:** "A famous quote." **End quote.**\n\n'
            "More content.\n\n"
            "> **Annotated summary:**<br>\n"
            "> Summary text.\n"
            "> **End annotated summary.**"
        )
        temp_markdown_file.write_text(content, encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        expected = (
            "# Chapter\n\n"
            "> **Introduction:**<br>\n"
            "> Introduction text.\n"
            "> **End annotated introduction.**\n\n"
            "Main content.\n\n"
            "> **Note:** A note about something. **End annotation.**\n\n"
            '> **Quote:** "A famous quote." **End quote.**\n\n'
            "More content.\n\n"
            "> **Summary:**<br>\n"
            "> Summary text.\n"
            "> **End annotated summary.**"
        )
        assert result == expected

    def test_markers_with_varying_whitespace(self, builder, temp_markdown_file):
        """Test that markers with different whitespace patterns are handled correctly."""
        content = (
            "Text.\n\n"
            ">**Annotation:** No space after >. **End annotation.**\n\n"
            ">  **Annotation:** Two spaces after >. **End annotation.**\n\n"
            "> **Annotation:** One space after >. **End annotation.**\n\n"
            ">\t**Annotation:** Tab after >. **End annotation.**"
        )
        temp_markdown_file.write_text(content, encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        expected = (
            "Text.\n\n"
            ">**Note:** No space after >. **End annotation.**\n\n"
            ">  **Note:** Two spaces after >. **End annotation.**\n\n"
            "> **Note:** One space after >. **End annotation.**\n\n"
            ">\t**Note:** Tab after >. **End annotation.**"
        )
        assert result == expected

    def test_markers_not_in_blockquotes_unchanged(self, builder, temp_markdown_file):
        """Test that markers not in blockquotes (no >) are left unchanged."""
        content = (
            "**Annotation:** This is not in a blockquote.\n\n"
            "> **Annotation:** This is in a blockquote. **End annotation.**\n\n"
            "Regular text with **Annotated introduction:** in the middle.\n\n"
            "> **Annotated summary:** This should change. **End annotated summary.**"
        )
        temp_markdown_file.write_text(content, encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        expected = (
            "**Annotation:** This is not in a blockquote.\n\n"
            "> **Note:** This is in a blockquote. **End annotation.**\n\n"
            "Regular text with **Annotated introduction:** in the middle.\n\n"
            "> **Summary:** This should change. **End annotated summary.**"
        )
        assert result == expected

    def test_empty_file(self, builder, temp_markdown_file):
        """Test that empty files are handled gracefully."""
        temp_markdown_file.write_text("", encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        assert result == ""

    def test_no_markers_to_replace(self, builder, temp_markdown_file):
        """Test that files without markers remain unchanged."""
        content = (
            "# Chapter Title\n\n"
            "This is regular text.\n\n"
            "> This is a regular blockquote.\n"
            "> It continues here.\n\n"
            "More regular text."
        )
        temp_markdown_file.write_text(content, encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        assert result == content

    def test_multiline_blockquote_annotations(self, builder, temp_markdown_file):
        """Test markers in multiline blockquote structures."""
        content = (
            "> **Annotated introduction:**<br>\n"
            "><br>\n"
            "> **Introduction**<br>\n"
            "> This is a detailed overview spanning multiple lines.\n"
            "> It continues here.<br>\n"
            "><br>\n"
            "> **Key Terms**<br>\n"
            "> *Term 1* - Definition<br>\n"
            "> *Term 2* - Definition<br>\n"
            "><br>\n"
            "> **End annotated introduction.**"
        )
        temp_markdown_file.write_text(content, encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        expected = (
            "> **Introduction:**<br>\n"
            "><br>\n"
            "> **Introduction**<br>\n"
            "> This is a detailed overview spanning multiple lines.\n"
            "> It continues here.<br>\n"
            "><br>\n"
            "> **Key Terms**<br>\n"
            "> *Term 1* - Definition<br>\n"
            "> *Term 2* - Definition<br>\n"
            "><br>\n"
            "> **End annotated introduction.**"
        )
        assert result == expected

    def test_unicode_content_with_markers(self, builder, temp_markdown_file):
        """Test marker replacement with Unicode content."""
        content = (
            "Introduction.\n\n"
            "> **Annotation:** 中文注释内容 (Chinese annotation). **End annotation.**\n\n"
            "> **Annotated summary:**<br>\n"
            "> العربية النص (Arabic text with émojis 🌍).\n"
            "> **End annotated summary.**\n\n"
            '> **Quote:** "Привет мир! 📚" (Russian quote) **End quote.**'
        )
        temp_markdown_file.write_text(content, encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        expected = (
            "Introduction.\n\n"
            "> **Note:** 中文注释内容 (Chinese annotation). **End annotation.**\n\n"
            "> **Summary:**<br>\n"
            "> العربية النص (Arabic text with émojis 🌍).\n"
            "> **End annotated summary.**\n\n"
            '> **Quote:** "Привет мир! 📚" (Russian quote) **End quote.**'
        )
        assert result == expected

    def test_case_sensitivity(self, builder, temp_markdown_file):
        """Test that marker replacement is case-sensitive."""
        content = (
            "> **annotation:** Lowercase should not be replaced. **End annotation.**\n\n"
            "> **Annotation:** Correct case should be replaced. **End annotation.**\n\n"
            "> **ANNOTATION:** Uppercase should not be replaced. **End annotation.**"
        )
        temp_markdown_file.write_text(content, encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        expected = (
            "> **annotation:** Lowercase should not be replaced. **End annotation.**\n\n"
            "> **Note:** Correct case should be replaced. **End annotation.**\n\n"
            "> **ANNOTATION:** Uppercase should not be replaced. **End annotation.**"
        )
        assert result == expected

    def test_partial_marker_matches(self, builder, temp_markdown_file):
        """Test that partial marker matches are not replaced."""
        content = (
            "> **Annotated:** Not a full marker. **End annotated.**\n\n"
            "> **Annotation:** Full marker. **End annotation.**\n\n"
            "> **Annotations:** Plural form. **End annotations.**"
        )
        temp_markdown_file.write_text(content, encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        expected = (
            "> **Annotated:** Not a full marker. **End annotated.**\n\n"
            "> **Note:** Full marker. **End annotation.**\n\n"
            "> **Annotations:** Plural form. **End annotations.**"
        )
        assert result == expected

    def test_markers_at_line_start_only(self, builder, temp_markdown_file):
        """Test that markers are only replaced when they follow the blockquote marker."""
        content = (
            "> Text before **Annotation:** should not be replaced.\n\n"
            "> **Annotation:** At start should be replaced. **End annotation.**\n\n"
            "> Some text. **Annotated introduction:** in middle should not be replaced."
        )
        temp_markdown_file.write_text(content, encoding="utf-8")

        builder.clean_start_markers(input_path=temp_markdown_file)

        result = temp_markdown_file.read_text(encoding="utf-8")
        expected = (
            "> Text before **Annotation:** should not be replaced.\n\n"
            "> **Note:** At start should be replaced. **End annotation.**\n\n"
            "> Some text. **Annotated introduction:** in middle should not be replaced."
        )
        assert result == expected
