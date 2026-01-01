"""Tests for book_writer parser module."""

from book_writer.parser import (
    compute_rubric_hash,
    parse_rubric,
)


class TestComputeRubricHash:
    """Tests for compute_rubric_hash function."""

    def test_hash_is_deterministic(self, temp_dir):
        """Test that the same content produces the same hash."""
        rubric_file = temp_dir / "rubric.md"
        content = "# Test Book\n\n# Chapter 1: Test\n"
        rubric_file.write_text(content)

        hash1 = compute_rubric_hash(rubric_file)
        hash2 = compute_rubric_hash(rubric_file)

        assert hash1 == hash2

    def test_different_content_different_hash(self, temp_dir):
        """Test that different content produces different hashes."""
        rubric1 = temp_dir / "rubric1.md"
        rubric2 = temp_dir / "rubric2.md"

        rubric1.write_text("# Book One")
        rubric2.write_text("# Book Two")

        hash1 = compute_rubric_hash(rubric1)
        hash2 = compute_rubric_hash(rubric2)

        assert hash1 != hash2

    def test_hash_is_sha256(self, temp_dir):
        """Test that hash is proper SHA256 format."""
        rubric_file = temp_dir / "rubric.md"
        rubric_file.write_text("content")

        hash_value = compute_rubric_hash(rubric_file)

        # SHA256 produces 64 hex characters
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)


class TestParseRubric:
    """Tests for parse_rubric function."""

    def test_parse_book_title(self, sample_rubric_file):
        """Test parsing book title."""
        outline = parse_rubric(sample_rubric_file)
        assert outline.title == "My Test Book"

    def test_parse_chapters(self, sample_rubric_file):
        """Test parsing chapters."""
        outline = parse_rubric(sample_rubric_file)
        assert len(outline.chapters) == 2

        # Check first chapter
        ch1 = outline.chapters[0]
        assert ch1.id == "0"
        assert ch1.number == 0
        assert ch1.title == "Introduction"

        # Check second chapter
        ch2 = outline.chapters[1]
        assert ch2.id == "1"
        assert ch2.number == 1
        assert ch2.title == "Advanced Topics"

    def test_parse_sections(self, sample_rubric_file):
        """Test parsing sections."""
        outline = parse_rubric(sample_rubric_file)
        ch1 = outline.chapters[0]

        assert len(ch1.sections) == 2
        assert ch1.sections[0].id == "0.0"
        assert "Opening Vignette" in ch1.sections[0].title
        assert ch1.sections[1].id == "0.1"
        assert "Core Concepts" in ch1.sections[1].title

    def test_parse_section_content(self, sample_rubric_file):
        """Test parsing section outline content."""
        outline = parse_rubric(sample_rubric_file)
        ch1 = outline.chapters[0]
        section = ch1.sections[0]

        assert section.outline_content is not None
        assert "Some intro content" in section.outline_content
        assert "opening story" in section.outline_content.lower()

    def test_parse_empty_rubric(self, temp_dir):
        """Test parsing an empty rubric."""
        rubric_file = temp_dir / "empty.md"
        rubric_file.write_text("")

        outline = parse_rubric(rubric_file)
        assert outline.title == "Untitled Book"
        assert outline.chapters == []

    def test_parse_rubric_with_only_title(self, temp_dir):
        """Test parsing rubric with only title."""
        rubric_file = temp_dir / "title_only.md"
        rubric_file.write_text("# My Only Title\n")

        outline = parse_rubric(rubric_file)
        assert outline.title == "My Only Title"
        assert outline.chapters == []

    def test_line_numbers_are_tracked(self, sample_rubric_file):
        """Test that line numbers are tracked."""
        outline = parse_rubric(sample_rubric_file)
        ch1 = outline.chapters[0]

        # Chapter and sections should have line numbers
        assert ch1.line_start >= 0
        assert ch1.line_end > ch1.line_start

        for section in ch1.sections:
            assert section.line_start >= 0
            assert section.line_end >= section.line_start

    def test_chapters_preserve_document_order(self, temp_dir):
        """Test chapters are parsed in document order."""
        rubric_file = temp_dir / "order.md"
        rubric_file.write_text("""# Test Book

## Third Chapter

### Section One

Content.

## First Chapter

### Section Two

Content.
""")

        outline = parse_rubric(rubric_file)
        assert [ch.title for ch in outline.chapters] == ["Third Chapter", "First Chapter"]
        assert [ch.id for ch in outline.chapters] == ["0", "1"]

    def test_sections_preserve_document_order(self, temp_dir):
        """Test sections are parsed in document order within a chapter."""
        rubric_file = temp_dir / "section_order.md"
        rubric_file.write_text("""# Test Book

## Chapter One

### Second Section

Content.

### First Section

Content.
""")

        outline = parse_rubric(rubric_file)
        section_titles = [section.title for section in outline.chapters[0].sections]
        section_ids = [section.id for section in outline.chapters[0].sections]
        assert section_titles == ["Second Section", "First Section"]
        assert section_ids == ["0.0", "0.1"]
