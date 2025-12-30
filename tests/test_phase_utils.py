from book_updater.phases.utils import extract_markdown_blocks


def test_extract_blocks_final_section_no_content():
    text = "## Chapter 1\nContent\n## Chapter 2"
    blocks = extract_markdown_blocks(text)
    assert len(blocks) == 2
    assert blocks[1].strip() == "## Chapter 2"


def test_extract_blocks_consecutive_headers():
    text = "## Chapter 1\n## Chapter 2\nMore\n"
    blocks = extract_markdown_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].strip() == "## Chapter 1"
    assert blocks[1].startswith("## Chapter 2")


def test_extract_blocks_preserves_preamble():
    text = "Intro line\n\n## Chapter 1\nBody\n"
    blocks = extract_markdown_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].strip() == "Intro line"
    assert blocks[1].startswith("## Chapter 1")


def test_extract_blocks_no_headers_returns_text():
    text = "Just text\nNo headers here.\n"
    blocks = extract_markdown_blocks(text)
    assert blocks == [text]
