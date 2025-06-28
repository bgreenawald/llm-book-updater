import pytest

from src.post_processors import (
    EnsureBlankLineProcessor,
    OrderQuoteAnnotationProcessor,
    RemoveTrailingWhitespaceProcessor,
    RemoveXmlTagsProcessor,
)


@pytest.fixture
def ensure_blank_line_processor():
    return EnsureBlankLineProcessor()


@pytest.fixture
def remove_xml_tags_processor():
    return RemoveXmlTagsProcessor()


@pytest.fixture
def remove_trailing_whitespace_processor():
    return RemoveTrailingWhitespaceProcessor()


@pytest.fixture
def order_quote_annotation_processor():
    return OrderQuoteAnnotationProcessor()


def test_ensure_blank_line_between_elements(ensure_blank_line_processor):
    llm_block = "Paragraph 1.\nParagraph 2."
    expected_output = "Paragraph 1.\n\nParagraph 2."
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


def test_no_blank_line_for_list_items(ensure_blank_line_processor):
    llm_block = "* Item 1\n* Item 2"
    expected_output = "* Item 1\n* Item 2"
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


def test_no_blank_line_for_multiline_quote(ensure_blank_line_processor):
    llm_block = "> Quote line 1\n> Quote line 2"
    expected_output = "> Quote line 1\n> Quote line 2"
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


def test_remove_xml_tags(remove_xml_tags_processor):
    llm_block = "<p>Some text.</p>\n<br>\n<item>Another item</item>"
    expected_output = "Some text.\n<br>\nAnother item"
    assert remove_xml_tags_processor.process("", llm_block) == expected_output


def test_remove_trailing_whitespace(remove_trailing_whitespace_processor):
    llm_block = "Line 1  \nLine 2\t\nLine 3"
    expected_output = "Line 1\nLine 2\nLine 3"
    assert remove_trailing_whitespace_processor.process("", llm_block) == expected_output


def test_blank_line_between_special_blocks(ensure_blank_line_processor):
    llm_block = "> **Quote:** A quote. **End quote.**\n> **Annotation:** An annotation. **End annotation.**"
    expected_output = "> **Quote:** A quote. **End quote.**\n\n> **Annotation:** An annotation. **End annotation.**"
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


def test_quote_block_surrounded_by_blank_lines(ensure_blank_line_processor):
    """Test that Quote blocks are properly surrounded by blank lines."""
    llm_block = "Some text.\n> **Quote:** This is a quote. **End quote.**\nMore text."
    expected_output = "Some text.\n\n> **Quote:** This is a quote. **End quote.**\n\nMore text."
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


def test_annotation_block_surrounded_by_blank_lines(ensure_blank_line_processor):
    """Test that Annotation blocks are properly surrounded by blank lines."""
    llm_block = "Some text.\n> **Annotation:** This is an annotation. **End annotation.**\nMore text."
    expected_output = "Some text.\n\n> **Annotation:** This is an annotation. **End annotation.**\n\nMore text."
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


def test_multiple_quote_blocks_with_blank_lines(ensure_blank_line_processor):
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
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


def test_quote_block_with_multiple_lines(ensure_blank_line_processor):
    """Test that Quote blocks are single-line and don't have internal blank lines."""
    llm_block = "Text before.\n> **Quote:** This is a single-line quote block. **End quote.**\nText after."
    expected_output = "Text before.\n\n> **Quote:** This is a single-line quote block. **End quote.**\n\nText after."
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


def test_annotation_block_with_multiple_lines(ensure_blank_line_processor):
    """Test that Annotation blocks are single-line and don't have internal blank lines."""
    llm_block = (
        "Text before.\n> **Annotation:** This is a single-line annotation block. **End annotation.**\nText after."
    )
    expected_output = (
        "Text before.\n\n> **Annotation:** This is a single-line annotation block. **End annotation.**\n\nText after."
    )
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


def test_regular_quotes_not_surrounded(ensure_blank_line_processor):
    """Test that regular quotes (not Quote/Annotation blocks) are not surrounded."""
    llm_block = "Some text.\n> This is a regular quote.\n> Not a special block.\nMore text."
    expected_output = "Some text.\n\n> This is a regular quote.\n> Not a special block.\n\nMore text."
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


def test_mixed_content_with_quotes_and_lists(ensure_blank_line_processor):
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
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


def test_empty_lines_preserved(ensure_blank_line_processor):
    """Test that existing empty lines are preserved."""
    llm_block = "Line 1.\n\nLine 2.\n\n\nLine 3."
    expected_output = "Line 1.\n\nLine 2.\n\n\nLine 3."
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


def test_quote_block_at_end(ensure_blank_line_processor):
    """Test Quote block at the end of content."""
    llm_block = "Some text.\n> **Quote:** Final quote. **End quote.**"
    expected_output = "Some text.\n\n> **Quote:** Final quote. **End quote.**"
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


def test_quote_block_at_beginning(ensure_blank_line_processor):
    """Test Quote block at the beginning of content."""
    llm_block = "> **Quote:** Opening quote. **End quote.**\nSome text."
    expected_output = "> **Quote:** Opening quote. **End quote.**\n\nSome text."
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


def test_multiline_quotes_still_work(ensure_blank_line_processor):
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
    assert ensure_blank_line_processor.process("", llm_block) == expected_output


# OrderQuoteAnnotationProcessor tests
def test_simple_reorder_annotation_before_quote(order_quote_annotation_processor):
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
    assert order_quote_annotation_processor.process("", llm_block) == expected_output


def test_simple_reorder_annotation_before_quote_with_newline(order_quote_annotation_processor):
    """Test reordering when annotation comes before quote."""
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
    assert order_quote_annotation_processor.process("", llm_block) == expected_output


def test_multiple_quotes_and_annotations(order_quote_annotation_processor):
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
    assert order_quote_annotation_processor.process("", llm_block) == expected_output


def test_separate_blocks_not_affected(order_quote_annotation_processor):
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
    assert order_quote_annotation_processor.process("", llm_block) == expected_output


def test_quotes_only_no_change(order_quote_annotation_processor):
    """Test that blocks with only quotes are not changed."""
    llm_block = (
        "Some text.\n> **Quote:** First quote. **End quote.**\n> **Quote:** Second quote. **End quote.**\nMore text."
    )
    expected_output = (
        "Some text.\n> **Quote:** First quote. **End quote.**\n> **Quote:** Second quote. **End quote.**\nMore text."
    )
    assert order_quote_annotation_processor.process("", llm_block) == expected_output


def test_annotations_only_no_change(order_quote_annotation_processor):
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
    assert order_quote_annotation_processor.process("", llm_block) == expected_output


def test_mixed_content_with_regular_quotes(order_quote_annotation_processor):
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
    assert order_quote_annotation_processor.process("", llm_block) == expected_output


def test_empty_input(order_quote_annotation_processor):
    """Test with empty input."""
    llm_block = ""
    expected_output = ""
    assert order_quote_annotation_processor.process("", llm_block) == expected_output


def test_no_quotes_or_annotations(order_quote_annotation_processor):
    """Test with no Quote or Annotation blocks."""
    llm_block = "First paragraph.\nSecond paragraph.\n> Regular quote.\nThird paragraph."
    expected_output = "First paragraph.\nSecond paragraph.\n> Regular quote.\nThird paragraph."
    assert order_quote_annotation_processor.process("", llm_block) == expected_output


def test_complex_mixed_scenario(order_quote_annotation_processor):
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
    assert order_quote_annotation_processor.process("", llm_block) == expected_output
