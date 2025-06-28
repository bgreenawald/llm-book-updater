import pytest
from src.post_processors import NoNewHeadersPostProcessor


@pytest.fixture
def processor():
    return NoNewHeadersPostProcessor()


def test_revert_converted_header(processor):
    original_block = "This is a test line."
    llm_block = "# This is a test line."
    expected_output = "This is a test line."
    assert processor.process(original_block, llm_block) == expected_output


def test_remove_new_header(processor):
    original_block = "This is a test line."
    llm_block = "This is a test line.\n# This is a new header."
    expected_output = "This is a test line."
    assert processor.process(original_block, llm_block) == expected_output


def test_no_change(processor):
    original_block = "# This is a test line."
    llm_block = "# This is a test line."
    expected_output = "# This is a test line."
    assert processor.process(original_block, llm_block) == expected_output


def test_mixed_changes(processor):
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
    assert processor.process(original_block, llm_block) == expected_output


def test_empty_input(processor):
    original_block = ""
    llm_block = ""
    expected_output = ""
    assert processor.process(original_block, llm_block) == expected_output


def test_llm_block_empty(processor):
    original_block = "Some content."
    llm_block = ""
    expected_output = ""
    assert processor.process(original_block, llm_block) == expected_output
