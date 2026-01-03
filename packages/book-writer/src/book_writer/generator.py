"""Core generation logic for book writing."""

import asyncio
from pathlib import Path
from typing import Any, Callable, Optional

from llm_core import AsyncOpenRouterClient, LlmModelError
from loguru import logger

from .models import (
    BookOutline,
    BookState,
    ChapterOutline,
    ChapterStatus,
    GenerationConfig,
    SectionOutline,
    SectionStatus,
)
from .prompts import build_identify_prompt, build_implement_prompt, build_section_prompt
from .state import StateManager

# Markers used by Phase 2 to indicate changes or no changes
NO_CHANGES_MARKER = "No Changes Recommended"
CHANGE_MARKER = "### Change"


def _feedback_indicates_no_changes(feedback: str) -> bool:
    """
    Check if the identify phase feedback indicates no changes are needed.

    Returns True only if:
    - The feedback contains "No Changes Recommended" AND
    - The feedback does NOT contain any "### Change" markers

    This prevents false positives where the model mentions "no changes recommended"
    for one aspect while still proposing changes for others.
    """
    has_no_changes_marker = NO_CHANGES_MARKER in feedback
    has_change_markers = CHANGE_MARKER in feedback

    # Only skip Phase 3 if explicitly no changes AND no actual change proposals
    return has_no_changes_marker and not has_change_markers


def get_chapter_filename(chapter: ChapterOutline) -> str:
    """Generate the filename for a chapter based on its ID.

    Args:
        chapter: The chapter outline to generate a filename for.

    Returns:
        A filename string in the format "chapter_XX.md" where XX is zero-padded.

    Raises:
        ValueError: If chapter.number is None and chapter.id is not a numeric string.
    """
    if chapter.number is not None:
        chapter_num = chapter.number
    else:
        if not chapter.id.isdigit():
            raise ValueError(
                f"Chapter ID '{chapter.id}' must be a numeric string when "
                f"chapter.number is None. Non-digit IDs are not supported for "
                f"filename generation to prevent collisions."
            )
        chapter_num = int(chapter.id)
    return f"chapter_{chapter_num:02d}.md"


class BookGenerator:
    """Orchestrates parallel chapter generation with sequential section processing."""

    def __init__(
        self,
        outline: BookOutline,
        client: AsyncOpenRouterClient,
        state_manager: StateManager,
        config: GenerationConfig,
        output_dir: Path,
        progress_callback: Optional[Callable] = None,
        max_sections_per_chapter: Optional[int] = None,
    ):
        self.outline = outline
        self.client = client
        self.state_manager = state_manager
        self.config = config
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        self.max_sections_per_chapter = max_sections_per_chapter

        # Build chapter lookup
        self._chapters: dict[str, ChapterOutline] = {}
        for chapter in outline.chapters:
            self._chapters[chapter.id] = chapter

    def _get_sections_for_chapter(self, chapter: ChapterOutline) -> list[SectionOutline]:
        if self.max_sections_per_chapter is None:
            return chapter.sections
        return chapter.sections[: self.max_sections_per_chapter]

    async def generate_book(
        self,
        state: BookState,
        chapters_to_process: Optional[list[str]] = None,
    ) -> BookState:
        """
        Generate all chapters in parallel.
        Each chapter processes sections sequentially.
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent_chapters)

        # Determine which chapters to process
        if chapters_to_process:
            chapter_ids = chapters_to_process
        else:
            chapter_ids = list(self._chapters.keys())

        # Create tasks for each chapter, preserving chapter_id mapping
        task_chapter_pairs = [
            (self._generate_chapter_with_semaphore(semaphore, state, chapter_id), chapter_id)
            for chapter_id in chapter_ids
            if chapter_id in self._chapters
        ]
        tasks = [task for task, _ in task_chapter_pairs]
        chapter_ids_list = [chapter_id for _, chapter_id in task_chapter_pairs]

        # Run all chapters in parallel (limited by semaphore)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions and log them with context
        exceptions: list[Exception] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                chapter_id = chapter_ids_list[idx]
                logger.error(
                    f"Chapter generation failed for chapter_id='{chapter_id}' (index={idx}): "
                    f"{type(result).__name__}: {result}",
                    exc_info=result,
                )
                exceptions.append(result)

        # Reload and return final state (even if there were exceptions)
        final_state = self.state_manager.load_state() or state

        # Raise aggregate error if any exceptions occurred (after state reload)
        if exceptions:
            if len(exceptions) == 1:
                # Single exception - re-raise it directly
                raise exceptions[0]
            else:
                # Multiple exceptions - use ExceptionGroup (Python 3.11+)
                raise ExceptionGroup(
                    f"Multiple chapter generation failures ({len(exceptions)} chapters failed)",
                    exceptions,
                )

        return final_state

    async def _generate_chapter_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        state: BookState,
        chapter_id: str,
    ) -> None:
        """Wrapper to limit concurrent chapter generation."""
        async with semaphore:
            await self._generate_chapter(state, chapter_id)

    async def _generate_chapter(
        self,
        state: BookState,
        chapter_id: str,
    ) -> None:
        """
        Generate a single chapter by processing sections sequentially.
        Stops on first failed section after retries.
        """
        chapter = self._chapters.get(chapter_id)
        if not chapter:
            return

        if chapter_id not in state.chapters:
            return

        chapter_state = state.chapters[chapter_id]

        # Skip if already completed
        if chapter_state.status == ChapterStatus.COMPLETED:
            self._notify_progress(chapter_id, None, "skipped", "Already completed")
            return

        # Mark chapter as in progress
        self.state_manager.mark_chapter_started(state, chapter_id)
        self._notify_progress(chapter_id, None, "started")

        # Track previously generated content for context
        previous_sections: list[tuple[str, str]] = []
        sections_to_process = self._get_sections_for_chapter(chapter)

        # First, load any already completed sections
        for section in sections_to_process:
            section_state = chapter_state.sections.get(section.id)
            if section_state and section_state.status == SectionStatus.COMPLETED:
                if section_state.generated_content:
                    previous_sections.append((section.title, section_state.generated_content))

        # Process each section sequentially
        for section in sections_to_process:
            section_state = chapter_state.sections.get(section.id)
            if not section_state:
                continue

            # Skip already completed sections
            if section_state.status == SectionStatus.COMPLETED:
                continue

            # Generate this section
            success, content = await self._generate_section(chapter, section, previous_sections, state)

            if success and content:
                previous_sections.append((section.title, content))
            else:
                # Section failed after retries - stop this chapter
                self._notify_progress(
                    chapter_id,
                    section.id,
                    "chapter_stopped",
                    f"Stopped after section {section.id} failed",
                )
                await self._write_partial_chapter(chapter_id, state)
                return

        # All sections completed
        self._notify_progress(chapter_id, None, "chapter_completed")
        await self._write_complete_chapter(chapter_id, state)

    async def _generate_section(
        self,
        chapter: ChapterOutline,
        section: SectionOutline,
        previous_sections: list[tuple[str, str]],
        state: BookState,
    ) -> tuple[bool, Optional[str]]:
        """
        Generate a single section through the three-phase pipeline.

        Phase 1 (Generate): Create initial content from rubric
        Phase 2 (Identify): Analyze content and identify refinement opportunities
        Phase 3 (Implement): Apply refinements to produce final content

        Returns (success, final_content).
        """
        # Mark as in progress
        self.state_manager.update_section(state, chapter.id, section.id, status=SectionStatus.IN_PROGRESS)

        # Phase 1: Generate initial content
        success, initial_content = await self._run_generation_phase(chapter, section, previous_sections, state)
        if not success or not initial_content:
            return False, None

        # Phase 2: Identify refinements
        success, feedback = await self._run_identify_phase(chapter.id, section.id, initial_content, state)
        if not success or not feedback:
            return False, None

        # Check if Phase 2 indicates no changes are needed
        final_content: Optional[str] = None
        if _feedback_indicates_no_changes(feedback):
            # Skip Phase 3 - use initial content as final
            self._notify_progress(chapter.id, section.id, "phase3_skipped", "No refinements needed")
            final_content = initial_content
        else:
            # Phase 3: Implement refinements
            success, temp_final = await self._run_implement_phase(
                chapter.id, section.id, initial_content, feedback, state
            )
            if not success or not temp_final:
                return False, None
            final_content = temp_final

        # Save final content to state
        self.state_manager.update_section(
            state,
            chapter.id,
            section.id,
            status=SectionStatus.COMPLETED,
            content=final_content,
            initial_content=initial_content,
            identify_feedback=feedback,
        )

        self._notify_progress(chapter.id, section.id, "completed")
        return True, final_content

    async def _run_phase_with_error_handling(
        self,
        phase_name: str,
        phase_number: int,
        chapter_id: str,
        section_id: str,
        messages: list[dict[str, str]],
        state: BookState,
        **state_update_kwargs: Any,
    ) -> tuple[bool, Optional[str]]:
        """
        Execute a phase with standardized error handling and progress reporting.

        Args:
            phase_name: Human-readable phase name (e.g., "Generate", "Identify", "Implement").
            phase_number: Phase number (1, 2, or 3).
            chapter_id: Chapter identifier.
            section_id: Section identifier.
            messages: LLM messages for this phase.
            state: Current book state.
            **state_update_kwargs: Additional kwargs to pass to update_section on failure
                (e.g., initial_content, identify_feedback for debugging).

        Returns:
            Tuple of (success, content). On failure, returns (False, None).
        """
        # Convert phase name to gerund form for progress notification
        # Generate -> generating, Identify -> identifying, Implement -> implementing
        action_verb = phase_name.lower()
        if action_verb.endswith("e"):
            action_verb = action_verb[:-1] + "ing"
        elif not action_verb.endswith("ing"):
            action_verb = action_verb + "ing"

        self._notify_progress(chapter_id, section_id, f"phase{phase_number}_{action_verb}")

        try:
            model = self.config.get_model_for_phase(phase_number)
            content = await self.client.generate(messages, model=model)
            self._notify_progress(chapter_id, section_id, f"phase{phase_number}_completed")
            return True, content

        except LlmModelError as e:
            self.state_manager.update_section(
                state,
                chapter_id,
                section_id,
                status=SectionStatus.FAILED,
                error=f"Phase {phase_number} ({phase_name}) failed: {e}",
                **state_update_kwargs,
            )
            self._notify_progress(chapter_id, section_id, f"phase{phase_number}_failed", str(e))
            return False, None

    async def _run_generation_phase(
        self,
        chapter: ChapterOutline,
        section: SectionOutline,
        previous_sections: list[tuple[str, str]],
        state: BookState,
    ) -> tuple[bool, Optional[str]]:
        """
        Phase 1: Generate initial section content from rubric.

        Returns (success, initial_content).
        """
        messages = build_section_prompt(
            section=section,
            chapter=chapter,
            outline=self.outline,
            previous_sections=previous_sections,
        )

        return await self._run_phase_with_error_handling(
            phase_name="Generate",
            phase_number=1,
            chapter_id=chapter.id,
            section_id=section.id,
            messages=messages,
            state=state,
        )

    async def _run_identify_phase(
        self,
        chapter_id: str,
        section_id: str,
        initial_content: str,
        state: BookState,
    ) -> tuple[bool, Optional[str]]:
        """
        Phase 2: Analyze content and identify refinement opportunities.

        Returns (success, feedback).
        """
        messages = build_identify_prompt(generated_content=initial_content)

        return await self._run_phase_with_error_handling(
            phase_name="Identify",
            phase_number=2,
            chapter_id=chapter_id,
            section_id=section_id,
            messages=messages,
            state=state,
            initial_content=initial_content,  # Preserve P1 output for debugging
        )

    async def _run_implement_phase(
        self,
        chapter_id: str,
        section_id: str,
        initial_content: str,
        feedback: str,
        state: BookState,
    ) -> tuple[bool, Optional[str]]:
        """
        Phase 3: Apply identified refinements to produce final content.

        Returns (success, final_content).
        """
        messages = build_implement_prompt(
            generated_content=initial_content,
            feedback=feedback,
        )

        return await self._run_phase_with_error_handling(
            phase_name="Implement",
            phase_number=3,
            chapter_id=chapter_id,
            section_id=section_id,
            messages=messages,
            state=state,
            initial_content=initial_content,  # Preserve P1 output for debugging
            identify_feedback=feedback,  # Preserve P2 output for debugging
        )

    async def _write_partial_chapter(
        self,
        chapter_id: str,
        state: BookState,
    ) -> None:
        """Write completed sections of a chapter to disk."""
        chapter = self._chapters.get(chapter_id)
        if not chapter:
            return

        chapter_state = state.chapters.get(chapter_id)
        if not chapter_state:
            return

        await self._write_chapter_file(chapter, chapter_state, partial=True)

    async def _write_complete_chapter(
        self,
        chapter_id: str,
        state: BookState,
    ) -> None:
        """Write complete chapter to disk."""
        chapter = self._chapters.get(chapter_id)
        if not chapter:
            return

        chapter_state = state.chapters.get(chapter_id)
        if not chapter_state:
            return

        await self._write_chapter_file(chapter, chapter_state, partial=False)

    async def _write_chapter_file(
        self,
        chapter: ChapterOutline,
        chapter_state,
        partial: bool = False,
    ) -> None:
        """Write chapter content to markdown file."""
        chapters_dir = self.output_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename using helper
        filename = get_chapter_filename(chapter)
        filepath = chapters_dir / filename

        # Build chapter content
        lines = []

        # Chapter heading
        lines.append(f"# {chapter.title}")

        lines.append("")

        if partial:
            lines.append("> **Note**: This chapter is incomplete due to generation errors.")
            lines.append("")

        # Add each section
        for section in self._get_sections_for_chapter(chapter):
            section_state = chapter_state.sections.get(section.id)
            if not section_state:
                continue

            lines.append(f"## {section.title}")
            lines.append("")

            if section_state.status == SectionStatus.COMPLETED:
                if section_state.generated_content:
                    lines.append(section_state.generated_content)
            elif section_state.status == SectionStatus.FAILED:
                lines.append(f"> **Generation failed**: {section_state.last_error}")
            else:
                lines.append("> *Section not yet generated*")

            lines.append("")

        # Write to file
        filepath.write_text("\n".join(lines), encoding="utf-8")

    def _notify_progress(
        self,
        chapter_id: str,
        section_id: Optional[str],
        status: str,
        message: Optional[str] = None,
    ) -> None:
        """Notify progress callback if set."""
        if self.progress_callback:
            self.progress_callback(chapter_id, section_id, status, message)


def combine_chapters(output_dir: Path, outline: BookOutline) -> Path:
    """Combine all chapter markdown files into single book.md."""
    chapters_dir = output_dir / "chapters"
    book_md = output_dir / "book.md"

    # Get all chapter files in order by iterating through the outline
    chapter_files = []

    for chapter in outline.chapters:
        chapter_file = chapters_dir / get_chapter_filename(chapter)
        if chapter_file.exists():
            chapter_files.append(chapter_file)

    # Combine into book.md
    with open(book_md, "w", encoding="utf-8") as out:
        # Add YAML frontmatter
        out.write("---\n")
        out.write(f'title: "{outline.title}"\n')
        out.write("author: AI-Assisted Draft\n")
        out.write("---\n\n")

        for chapter_file in chapter_files:
            with open(chapter_file, encoding="utf-8") as f:
                out.write(f.read())
            out.write("\n\n---\n\n")

    return book_md
