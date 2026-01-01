"""Parser for rubric markdown files."""

import hashlib
from pathlib import Path

from .models import BookOutline, ChapterOutline, SectionOutline


def compute_rubric_hash(rubric_path: Path) -> str:
    """Compute SHA256 hash of rubric file for change detection."""
    content = rubric_path.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def parse_rubric(rubric_path: Path) -> BookOutline:
    """Parse the complete rubric markdown into structured outline."""
    content = rubric_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Extract book title from first H1 or use default
    title = "Untitled Book"
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    chapters = []
    i = 0
    chapter_index = 0
    while i < len(lines):
        if lines[i].startswith("## "):
            chapter, i = _parse_chapter(lines, i, chapter_index)
            chapters.append(chapter)
            chapter_index += 1
        else:
            i += 1

    return BookOutline(title=title, chapters=chapters)


def _parse_chapter(lines: list[str], start: int, chapter_index: int) -> tuple[ChapterOutline, int]:
    """Parse a single chapter from the lines starting at start index."""
    title_line = lines[start]
    title = title_line[3:].strip()

    line_start = start
    i = start + 1
    sections = []
    preamble_lines: list[str] = []
    section_index = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith("## "):
            break

        if line.startswith("### "):
            section, i = _parse_section(lines, i, chapter_index, section_index, preamble_lines)
            sections.append(section)
            section_index += 1
            preamble_lines = []
            continue

        preamble_lines.append(line)
        i += 1

    return (
        ChapterOutline(
            id=str(chapter_index),
            number=chapter_index,
            title=title,
            sections=sections,
            line_start=line_start,
            line_end=i - 1,
        ),
        i,
    )


def _parse_section(
    lines: list[str],
    start: int,
    chapter_index: int,
    section_index: int,
    preamble_lines: list[str],
) -> tuple[SectionOutline, int]:
    """Parse a single section (### heading) and its content."""
    title_line = lines[start]
    full_title = title_line[4:].strip()
    section_id = f"{chapter_index}.{section_index}"

    line_start = start
    i = start + 1

    content_lines = list(preamble_lines)
    while i < len(lines):
        line = lines[i]
        if line.startswith("## ") or line.startswith("### "):
            break
        content_lines.append(line)
        i += 1

    outline_content = "\n".join(content_lines).strip()

    return (
        SectionOutline(
            id=section_id,
            title=full_title,
            heading_level=3,
            outline_content=outline_content,
            line_start=line_start,
            line_end=i - 1,
        ),
        i,
    )
