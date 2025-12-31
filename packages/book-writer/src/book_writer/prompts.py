"""Prompt templates for LLM generation."""

from .models import ChapterOutline, SectionOutline

SYSTEM_PROMPT = """You are an expert author writing a book titled "{book_title}".

Your writing style should be:
- Authoritative but accessible
- No jargon without explanation
- Concrete examples over abstract theory
- Acknowledge complexity without drowning in it
- Include occasional dry humor where appropriate
- Not academic; not dumbed-down

IMPORTANT FORMATTING RULES:
- Write in flowing prose with proper paragraphs
- Use markdown formatting appropriately (headers for subsections, bold for emphasis, etc.)
- Do NOT include the section heading itself (it will be added automatically)
- Focus only on the content described in the outline
- Maintain consistency with previously written sections
- If the outline includes ### subheadings, incorporate those naturally into your writing

IMPORTANT CONTENT RULES:
- Length should be section appropriate, based on the complexity of the section content
- Make sure content is thoroughly covered *without* being overly verbose
- The final result should be lean but complete
"""

SECTION_PROMPT = """## Current Task
Write the content for section "{section_title}" of {chapter_type} {chapter_id}: {chapter_title}.

## Chapter Goals
{chapter_goals}

## Section Outline (what to cover)
{section_outline}

## Previously Written Sections in This Chapter
{previous_sections}

## Instructions
1. Write ONLY this section's content based on the outline above
2. Build naturally on the previous sections (if any)
3. Match the tone and depth established in earlier sections
4. Follow the outline structure (the ### headings indicate subsections to cover)
5. Do NOT repeat content from previous sections
6. Do NOT include the section heading itself (e.g., don't start with "## 1.1 Core Idea...")
7. Start directly with the content

Begin writing the section content now:
"""

FIRST_SECTION_PROMPT = """## Current Task
Write the content for section "{section_title}" of {chapter_type} {chapter_id}: {chapter_title}.

This is the FIRST section of the chapter, so establish the chapter's tone and themes.

## Chapter Goals
{chapter_goals}

## Section Outline (what to cover)
{section_outline}

## Instructions
1. Write ONLY this section's content based on the outline above
2. This is the opening section - hook the reader and establish context
3. Follow the outline structure (the ### headings indicate subsections to cover)
4. Do NOT include the section heading itself (e.g., don't start with "## 1.1 Core Idea...")
5. Start directly with the content

Begin writing the section content now:
"""


def build_section_prompt(
    section: SectionOutline,
    chapter: ChapterOutline,
    book_title: str,
    previous_sections: list[tuple[str, str]],  # [(section_title, content), ...]
) -> list[dict]:
    """Build the complete messages array for section generation."""
    system_msg = SYSTEM_PROMPT.format(book_title=book_title)

    # Determine chapter type
    if chapter.id == "preface":
        chapter_type = "Preface"
        chapter_display_id = ""
    elif chapter.id.startswith("appendix_"):
        chapter_type = "Appendix"
        chapter_display_id = chapter.id.replace("appendix_", "").upper()
    else:
        chapter_type = "Chapter"
        chapter_display_id = chapter.id

    if not previous_sections:
        user_msg = FIRST_SECTION_PROMPT.format(
            section_title=section.title,
            chapter_type=chapter_type,
            chapter_id=chapter_display_id,
            chapter_title=chapter.title,
            chapter_goals=chapter.goals or "Not specified",
            section_outline=section.outline_content,
        )
    else:
        # Format previous sections
        prev_text = "\n\n---\n\n".join([f"### {title}\n\n{content}" for title, content in previous_sections])
        user_msg = SECTION_PROMPT.format(
            section_title=section.title,
            chapter_type=chapter_type,
            chapter_id=chapter_display_id,
            chapter_title=chapter.title,
            chapter_goals=chapter.goals or "Not specified",
            section_outline=section.outline_content,
            previous_sections=prev_text,
        )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
