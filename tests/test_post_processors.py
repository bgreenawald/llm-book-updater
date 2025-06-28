import pytest
from src.post_processors import NoNewHeadersPostProcessor, RevertRemovedBlockLines


@pytest.fixture
def no_new_headers_processor():
    return NoNewHeadersPostProcessor()


@pytest.fixture
def revert_removed_block_lines_processor():
    return RevertRemovedBlockLines()


def test_revert_converted_header(no_new_headers_processor):
    original_block = "This is a test line."
    llm_block = "# This is a test line."
    expected_output = "This is a test line."
    assert no_new_headers_processor.process(original_block, llm_block) == expected_output


def test_remove_new_header(no_new_headers_processor):
    original_block = "This is a test line."
    llm_block = "This is a test line.\n# This is a new header."
    expected_output = "This is a test line."
    assert no_new_headers_processor.process(original_block, llm_block) == expected_output


def test_no_change(no_new_headers_processor):
    original_block = "# This is a test line."
    llm_block = "# This is a test line."
    expected_output = "# This is a test line."
    assert no_new_headers_processor.process(original_block, llm_block) == expected_output


def test_mixed_changes(no_new_headers_processor):
    original_block = (
        "First line.\n"
        "Second line that will be converted.\n"
        "# An existing header."
    )
    llm_block = (
        "First line.\n"
        "# Second line that will be converted.\n"
        "# An existing header.\n"
        "## A completely new header."
    )
    expected_output = (
        "First line.\n"
        "Second line that will be converted.\n"
        "# An existing header."
    )
    assert no_new_headers_processor.process(original_block, llm_block) == expected_output


def test_empty_input(no_new_headers_processor):
    original_block = ""
    llm_block = ""
    expected_output = ""
    assert no_new_headers_processor.process(original_block, llm_block) == expected_output


def test_llm_block_empty(no_new_headers_processor):
    original_block = "Some content."
    llm_block = ""
    expected_output = ""
    assert no_new_headers_processor.process(original_block, llm_block) == expected_output


def test_restore_single_removed_block_line(revert_removed_block_lines_processor):
    original_block = "First line.\n> This is a block quote.\nSecond line."
    llm_block = "First line.\nSecond line."
    expected_output = "First line.\n> This is a block quote.\nSecond line."
    assert revert_removed_block_lines_processor.process(original_block, llm_block) == expected_output


def test_restore_multiple_removed_block_lines(revert_removed_block_lines_processor):
    original_block = "First line.\n> Quote 1.\n> Quote 2.\nSecond line."
    llm_block = "First line.\nSecond line."
    expected_output = "First line.\n> Quote 1.\n> Quote 2.\nSecond line."
    assert revert_removed_block_lines_processor.process(original_block, llm_block) == expected_output


def test_no_block_lines_removed(revert_removed_block_lines_processor):
    original_block = "First line.\n> Quote 1.\nSecond line."
    llm_block = "First line.\n> Quote 1.\nSecond line."
    expected_output = "First line.\n> Quote 1.\nSecond line."
    assert revert_removed_block_lines_processor.process(original_block, llm_block) == expected_output


def test_regular_lines_removed(revert_removed_block_lines_processor):
    original_block = "First line.\nSecond line.\n> Quote 1."
    llm_block = "> Quote 1."
    expected_output = "> Quote 1."
    assert revert_removed_block_lines_processor.process(original_block, llm_block) == expected_output


def test_mixed_content_with_removed_block_lines(revert_removed_block_lines_processor):
    original_block = "Intro.\n> Important quote.\nSome text.\n> Another quote.\nOutro."
    llm_block = "Intro.\nSome text.\nOutro."
    expected_output = "Intro.\n> Important quote.\nSome text.\n> Another quote.\nOutro."
    assert revert_removed_block_lines_processor.process(original_block, llm_block) == expected_output


def test_restore_consecutive_removed_block_lines(revert_removed_block_lines_processor):
    original_block = "Line 1\n> Quote 1\n> Quote 2\n> Quote 3\nLine 2"
    llm_block = "Line 1\nLine 2"
    expected_output = "Line 1\n> Quote 1\n> Quote 2\n> Quote 3\nLine 2"
    assert revert_removed_block_lines_processor.process(original_block, llm_block) == expected_output


def test_restore_removed_first_line_block_comment(revert_removed_block_lines_processor):
    original_block = "> This is the first line.\nSecond line."
    llm_block = "Second line."
    expected_output = "> This is the first line.\nSecond line."
    assert revert_removed_block_lines_processor.process(original_block, llm_block) == expected_output


def test_restore_removed_last_line_block_comment(revert_removed_block_lines_processor):
    original_block = "First line.\n> This is the last line."
    llm_block = "First line."
    expected_output = "First line.\n> This is the last line."
    assert revert_removed_block_lines_processor.process(original_block, llm_block) == expected_output


def test_mixed_input_empty_llm_output_restores_only_blocklines(revert_removed_block_lines_processor):
    original_block = "This is a regular line.\n> This is a block quote.\nAnother regular line."
    llm_block = ""
    expected_output = "> This is a block quote."
    assert revert_removed_block_lines_processor.process(original_block, llm_block) == expected_output
