"""Parser for rubric markdown files."""

import hashlib
import re
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
        if line.startswith("# ") and not line.startswith("# Part"):
            # Check if it's a chapter heading
            if not re.match(r"^# Chapter \d+:", line):
                title = line[2:].strip()
                break

    # Parse the document structure
    chapters = []
    appendices = []
    preface = None
    parts = []
    final_notes = None

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect Part markers
        if line.startswith("# Part "):
            parts.append(line[2:].strip())
            i += 1
            continue

        # Detect Preface
        if line.startswith("# Preface"):
            preface, i = _parse_chapter(lines, i, "preface")
            continue

        # Detect Chapter
        chapter_match = re.match(r"^# Chapter (\d+):\s*(.+)$", line)
        if chapter_match:
            chapter_num = int(chapter_match.group(1))
            chapter, i = _parse_chapter(lines, i, str(chapter_num), chapter_num)
            chapters.append(chapter)
            continue

        # Detect Appendix
        appendix_match = re.match(r"^# Appendix ([A-Z]):\s*(.+)$", line)
        if appendix_match:
            appendix_id = f"appendix_{appendix_match.group(1).lower()}"
            appendix, i = _parse_chapter(lines, i, appendix_id)
            appendices.append(appendix)
            continue

        # Detect Final Notes section
        if line.startswith("# Final Notes"):
            final_notes, i = _extract_until_next_h1(lines, i + 1)
            continue

        i += 1

    return BookOutline(
        title=title,
        preface=preface,
        parts=parts,
        chapters=chapters,
        appendices=appendices,
        final_notes=final_notes,
    )


def _parse_chapter(
    lines: list[str], start: int, chapter_id: str, chapter_num: int | None = None
) -> tuple[ChapterOutline, int]:
    """Parse a single chapter from the lines starting at start index."""
    # Extract chapter title from the H1 line
    title_line = lines[start]
    if ":" in title_line:
        title = title_line.split(":", 1)[1].strip()
    else:
        title = title_line[2:].strip()  # Remove "# " prefix

    line_start = start
    i = start + 1

    # Find chapter goals if present
    goals = None
    sections = []
    summary_box = None

    while i < len(lines):
        line = lines[i]

        # Stop at next H1 (new chapter/section)
        if line.startswith("# "):
            break

        # Detect Chapter Goals
        if line.startswith("## Chapter Goals"):
            goals, i = _extract_until_next_h2(lines, i + 1)
            continue

        # Detect Summary Box
        if line.startswith("> ") and "Summary" in lines[i - 1] if i > 0 else False:
            summary_box = line[2:].strip()
            i += 1
            continue

        # Detect Section (## heading)
        if line.startswith("## ") and not line.startswith("## Chapter Goals"):
            section, i = _parse_section(lines, i, chapter_id)
            sections.append(section)
            continue

        i += 1

    return (
        ChapterOutline(
            id=chapter_id,
            number=chapter_num,
            title=title,
            goals=goals,
            sections=sections,
            summary_box=summary_box,
            line_start=line_start,
            line_end=i - 1,
        ),
        i,
    )


def _parse_section(lines: list[str], start: int, chapter_id: str) -> tuple[SectionOutline, int]:
    """Parse a single section (## heading) and its content."""
    title_line = lines[start]
    full_title = title_line[3:].strip()  # Remove "## " prefix

    # Extract section ID from title if present (e.g., "1.1 Core Idea: ...")
    section_id = _extract_section_id(full_title, chapter_id)

    line_start = start
    i = start + 1

    # Collect all content until next ## or #
    content_lines = []
    while i < len(lines):
        line = lines[i]
        if line.startswith("## ") or line.startswith("# "):
            break
        content_lines.append(line)
        i += 1

    outline_content = "\n".join(content_lines).strip()

    return (
        SectionOutline(
            id=section_id,
            title=full_title,
            heading_level=2,
            outline_content=outline_content,
            line_start=line_start,
            line_end=i - 1,
        ),
        i,
    )


def _extract_section_id(title: str, chapter_id: str) -> str:
    """Extract section ID like '1.1' from title, or generate one."""
    # Try to match patterns like "1.1 Core Idea" or "1.1: Core Idea"
    match = re.match(r"^(\d+\.\d+)\s*[:.]?\s*", title)
    if match:
        return match.group(1)

    # Try to match patterns like "Opening Vignette" -> use chapter_id + title hash
    # Generate a simple ID based on title
    clean_title = re.sub(r"[^a-zA-Z0-9]", "_", title.lower())[:30]
    return f"{chapter_id}.{clean_title}"


def _extract_until_next_h1(lines: list[str], start: int) -> tuple[str, int]:
    """Extract content until the next H1 heading."""
    content_lines = []
    i = start
    while i < len(lines):
        if lines[i].startswith("# "):
            break
        content_lines.append(lines[i])
        i += 1
    return "\n".join(content_lines).strip(), i


def _extract_until_next_h2(lines: list[str], start: int) -> tuple[str, int]:
    """Extract content until the next H2 or H1 heading."""
    content_lines = []
    i = start
    while i < len(lines):
        if lines[i].startswith("## ") or lines[i].startswith("# "):
            break
        content_lines.append(lines[i])
        i += 1
    return "\n".join(content_lines).strip(), i
