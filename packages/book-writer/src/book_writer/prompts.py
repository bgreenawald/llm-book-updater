"""Prompt templates for LLM generation."""

from .models import ChapterOutline, SectionOutline

# =============================================================================
# Phase 2: Identify Refinements
# =============================================================================

IDENTIFY_SYSTEM_PROMPT = """You are an expert literary analyst and editorial consultant.
Your role is to analyze a generated text passage and identify specific opportunities for refinement.

**Important:** You are NOT making the changes yourself. You are creating a detailed change list
that another editor will use to apply the refinements.
Your analysis must be precise, actionable, and well-reasoned.
"""

IDENTIFY_USER_PROMPT = """## Generated Section
{generated_content}

## Your Task: Identify Opportunities for Refinement

Carefully analyze the passage and identify specific, targeted opportunities for enhancement.
For each opportunity you find, provide:

1. **Location**: The specific sentence, phrase, or paragraph where the change should occur.
2. **Type**: Which category of refinement this represents (clarity, voice, flow, accuracy, etc.).
3. **Current Text**: The exact text that should be modified.
4. **Proposed Change**: What the text should become (or what should be added).
5. **Rationale**: Why this change improves the passage.

## Output Format

Structure your output as a numbered list of proposed changes. For each change:

```
### Change [N]: [Type]

**Location:** [Where in the passage]

**Current text:** "[Exact text to be modified]"

**Proposed change:** "[What it should become]"

**Rationale:** [Brief explanation of why this improves the passage]
```

If no changes are needed, respond with:

```
### No Changes Recommended

The passage is well-crafted. No refinements are needed.
```

## Guidelines

* **Be Selective**: Only identify changes that genuinely improve the text. Quality over quantity.
* **Be Specific**: Provide exact text references so the implementing editor can make precise changes.
* **Preserve Length**: The passage length is intentional. Proposed changes should not significantly increase length.
"""


# =============================================================================
# Phase 3: Implement Refinements
# =============================================================================

IMPLEMENT_SYSTEM_PROMPT = """You are a skilled editor responsible for applying a set of
pre-identified changes to a text passage. A literary analyst has already reviewed the passage
and identified specific opportunities for refinement.
Your job is to apply these changes accurately and produce the final, polished version.
"""

IMPLEMENT_USER_PROMPT = """## Current Section
{generated_content}

## Identified Refinements
{feedback}

## Your Task

Apply each proposed change carefully and produce the final, refined passage.

## Guidelines for Applying Changes

1. **Apply Changes Precisely**: Use the exact proposed text when specified. Match the location carefully.

2. **Maintain Consistency**: Ensure that applied changes flow naturally with the surrounding text.
Make minor adjustments to transitions if needed for grammatical correctness.

3. **Preserve What's Not Changed**: Text that is not mentioned in the change list
should remain exactly as it appears in the Current Section.

4. **Handle "No Changes" Gracefully**: If the refinements indicate no changes are needed,
return the Current Section unchanged.

## Output Requirements

* Return **only** the final, polished passage.
* Do **not** include any introductory text, explanations, reports on your changes,
or any other metadata in your response.
* Your output must begin directly with the finalized content.
"""

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


def build_identify_prompt(generated_content: str) -> list[dict]:
    """Build the messages array for the identify refinements phase."""
    user_msg = IDENTIFY_USER_PROMPT.format(generated_content=generated_content)

    return [
        {"role": "system", "content": IDENTIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def build_implement_prompt(generated_content: str, feedback: str) -> list[dict]:
    """Build the messages array for the implement refinements phase."""
    user_msg = IMPLEMENT_USER_PROMPT.format(
        generated_content=generated_content,
        feedback=feedback,
    )

    return [
        {"role": "system", "content": IMPLEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
