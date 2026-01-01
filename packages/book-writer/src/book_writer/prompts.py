"""Prompt templates for LLM generation."""

from .models import ChapterOutline, SectionOutline

SYSTEM_PROMPT = """You are writing a book in a series of non-fiction books for curious, intelligent adults
who want genuine understanding of common topic areas.

The book you're writing is titled "{book_title}".

## Your Task

You will receive a section script: a sequence of instructions that define what to write and in what order.
Your job is to execute these instructions as clear, engaging prose that flows naturally.

The script handles structure and pedagogy. You handle voice, flow, readability, and additional content.
Imagine the script as a skeleton, and you are the flesh and blood that brings it to life.

## Reader Profile

Your reader has surface familiarity but not structural understanding. They've encountered the terminology
through news, conversation, and life—but couldn't confidently explain the underlying mechanics.
They know the words; you're providing the mental models.

## Voice and Style

- Authoritative but warm—a knowledgeable friend explaining something they find genuinely interesting
- Concrete before abstract; earn generalizations through specific examples
- Occasional dry humor where it lands naturally—never forced
- Not academic; not dumbed-down
- Your genuine fascination with the material should come through
- Respect the reader's intelligence while being patient with complexity

## Executing Scripts

Follow the script's instructions in order. Common elements you'll encounter:

- **Concepts**: Introduce clearly using the provided definition and anchor. The anchor is essential—it's what
  makes abstractions land. Bold the term on first use.
- **Misconceptions**: Address directly but without condescension. Pattern: "You might assume [X]—that's intuitive
  because [reason]. But [correction]." The reader should feel smart for understanding the correction,
  not dumb for having held the belief.
- **Callbacks**: Make the reader actively reconstruct the earlier concept. Don't just mention it—prompt retrieval:
  "Remember how we said [X]? That's exactly what's happening here..."
- **Friction**: Pose the question in a way that creates genuine pause. Let it breathe before resolving
  (if resolution is immediate) or note that we'll return to it.
- **Examples**: Bring these to life. They're not illustrations of the concept—they're how the concept becomes real.
- **Synthesis**: Reveal connections between concepts. Show structure, not just summary.

## Execution Guidelines

- Execute instructions in order, flowing naturally between them
- Script instructions are guidance, not text to copy verbatim
- Vary rhythm—sentence length, paragraph length
- Match depth to importance; load-bearing ideas get more space
- Trust the script's choices; your job is to make them sing

## Formatting

- Flowing prose with proper paragraphs
- **Bold** for key terms on first introduction
- *Italics* sparingly for emphasis
- Avoid bullet points unless genuinely necessary (rarely)
- Do NOT include the section heading (added automatically)
"""


SECTION_PROMPT = """## Section to Write

**Chapter {chapter_id}: {chapter_title}**
**Section {section_id}: {section_title}**

## Instructions
1. Write ONLY this section's content based on the section script below
2. Build naturally on the previous sections
3. Match the tone and depth established in earlier sections
4. Follow the outline structure (the ### headings indicate subsections to cover)
5. Do NOT repeat content from previous sections, unless instructed to do so in the section script
6. Do NOT include the section heading itself (e.g., don't start with "## 1.1 Core Idea...")
7. Start directly with the content

## Previously Written Sections in This Chapter
<START OF PREVIOUS SECTIONS>
{previous_sections}
<END OF PREVIOUS SECTIONS>

## Section Script
<START OF SECTION SCRIPT>
{script}
<END OF SECTION SCRIPT>

---

Begin writing the section:
"""


FIRST_SECTION_PROMPT = """## Section to Write

**Chapter {chapter_id}: {chapter_title}**
**Section {section_id}: {section_title}**

This is the FIRST section of the chapter, so establish the chapter's tone and themes.

## Instructions
1. Write ONLY this section's content based on the section script below
2. This is the opening section - hook the reader and establish context
3. Follow the outline structure (the ### headings indicate subsections to cover)
4. Do NOT include the section heading itself (e.g., don't start with "## 1.1 Core Idea...")
5. Start directly with the content

## Section Script
<START OF SECTION SCRIPT>
{script}
<END OF SECTION SCRIPT>
---

Begin writing the section:
"""

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
<START OF GENERATED SECTION>
{generated_content}
<END OF GENERATED SECTION>

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
<START OF GENERATED SECTION>
{generated_content}
<END OF GENERATED SECTION>

## Identified Refinements
<START OF IDENTIFIED REFINEMENTS>
{feedback}
<END OF IDENTIFIED REFINEMENTS>

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
