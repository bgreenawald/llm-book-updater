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

    def test_parse_parts(self, sample_rubric_file):
        """Test parsing parts."""
        outline = parse_rubric(sample_rubric_file)
        assert len(outline.parts) == 1
        assert "Part I: Foundations" in outline.parts[0]

    def test_parse_chapters(self, sample_rubric_file):
        """Test parsing chapters."""
        outline = parse_rubric(sample_rubric_file)
        assert len(outline.chapters) == 2

        # Check first chapter
        ch1 = outline.chapters[0]
        assert ch1.id == "1"
        assert ch1.number == 1
        assert ch1.title == "Introduction"

        # Check second chapter
        ch2 = outline.chapters[1]
        assert ch2.id == "2"
        assert ch2.number == 2
        assert ch2.title == "Advanced Topics"

    def test_parse_chapter_goals(self, sample_rubric_file):
        """Test parsing chapter goals."""
        outline = parse_rubric(sample_rubric_file)
        ch1 = outline.chapters[0]
        assert ch1.goals is not None
        assert "Introduce core concepts" in ch1.goals

    def test_parse_sections(self, sample_rubric_file):
        """Test parsing sections."""
        outline = parse_rubric(sample_rubric_file)
        ch1 = outline.chapters[0]

        assert len(ch1.sections) == 2
        assert ch1.sections[0].id == "1.1"
        assert "Opening Vignette" in ch1.sections[0].title
        assert ch1.sections[1].id == "1.2"
        assert "Core Concepts" in ch1.sections[1].title

    def test_parse_section_content(self, sample_rubric_file):
        """Test parsing section outline content."""
        outline = parse_rubric(sample_rubric_file)
        ch1 = outline.chapters[0]
        section = ch1.sections[0]

        assert section.outline_content is not None
        assert "opening story" in section.outline_content.lower()

    def test_parse_appendices(self, sample_rubric_file):
        """Test parsing appendices."""
        outline = parse_rubric(sample_rubric_file)
        assert len(outline.appendices) == 1

        appendix = outline.appendices[0]
        assert appendix.id == "appendix_a"
        assert "Reference Guide" in appendix.title

    def test_parse_final_notes(self, sample_rubric_file):
        """Test parsing final notes."""
        outline = parse_rubric(sample_rubric_file)
        assert outline.final_notes is not None
        assert "guidance" in outline.final_notes.lower()

    def test_parse_empty_rubric(self, temp_dir):
        """Test parsing an empty rubric."""
        rubric_file = temp_dir / "empty.md"
        rubric_file.write_text("")

        outline = parse_rubric(rubric_file)
        assert outline.title == "Untitled Book"
        assert outline.chapters == []
        assert outline.appendices == []

    def test_parse_rubric_with_only_title(self, temp_dir):
        """Test parsing rubric with only title."""
        rubric_file = temp_dir / "title_only.md"
        rubric_file.write_text("# My Only Title\n")

        outline = parse_rubric(rubric_file)
        assert outline.title == "My Only Title"
        assert outline.chapters == []

    def test_parse_preface(self, temp_dir):
        """Test parsing preface section."""
        rubric_file = temp_dir / "with_preface.md"
        rubric_file.write_text("""# Book Title

# Preface

## Welcome

This is the preface content.

# Chapter 1: First Chapter

## 1.1 Section

Content here.
""")

        outline = parse_rubric(rubric_file)
        assert outline.preface is not None
        assert outline.preface.id == "preface"
        assert len(outline.preface.sections) == 1

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

    def test_section_id_extraction(self, temp_dir):
        """Test section ID extraction from different formats."""
        rubric_file = temp_dir / "sections.md"
        rubric_file.write_text("""# Test Book

# Chapter 1: Test

## 1.1 Standard Format

Content.

## 2.3: With Colon

Content.

## Custom Section Title

Content.
""")

        outline = parse_rubric(rubric_file)
        ch1 = outline.chapters[0]

        assert ch1.sections[0].id == "1.1"
        assert ch1.sections[1].id == "2.3"
        # Third section should have generated ID
        assert ch1.sections[2].id.startswith("1.")

    def test_multiple_appendices(self, temp_dir):
        """Test parsing multiple appendices."""
        rubric_file = temp_dir / "appendices.md"
        rubric_file.write_text("""# Test Book

# Appendix A: First

## A.1 Section

Content.

# Appendix B: Second

## B.1 Section

Content.

# Appendix C: Third

## C.1 Section

Content.
""")

        outline = parse_rubric(rubric_file)
        assert len(outline.appendices) == 3
        assert outline.appendices[0].id == "appendix_a"
        assert outline.appendices[1].id == "appendix_b"
        assert outline.appendices[2].id == "appendix_c"
