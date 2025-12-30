"""Core generation logic for book writing."""

import asyncio
from pathlib import Path
from typing import Callable, Optional

from .models import (
    BookOutline,
    BookState,
    ChapterOutline,
    ChapterStatus,
    GenerationConfig,
    SectionOutline,
    SectionStatus,
)
from .openrouter import OpenRouterClient, OpenRouterError
from .prompts import build_section_prompt
from .state import StateManager


class BookGenerator:
    """Orchestrates parallel chapter generation with sequential section processing."""

    def __init__(
        self,
        outline: BookOutline,
        client: OpenRouterClient,
        state_manager: StateManager,
        config: GenerationConfig,
        output_dir: Path,
        progress_callback: Optional[Callable] = None,
    ):
        self.outline = outline
        self.client = client
        self.state_manager = state_manager
        self.config = config
        self.output_dir = output_dir
        self.progress_callback = progress_callback

        # Build chapter lookup
        self._chapters: dict[str, ChapterOutline] = {}
        if outline.preface:
            self._chapters[outline.preface.id] = outline.preface
        for chapter in outline.chapters:
            self._chapters[chapter.id] = chapter
        for appendix in outline.appendices:
            self._chapters[appendix.id] = appendix

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

        # Create tasks for each chapter
        tasks = [
            self._generate_chapter_with_semaphore(semaphore, state, chapter_id)
            for chapter_id in chapter_ids
            if chapter_id in self._chapters
        ]

        # Run all chapters in parallel (limited by semaphore)
        await asyncio.gather(*tasks, return_exceptions=True)

        # Reload and return final state
        return self.state_manager.load_state() or state

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

        # First, load any already completed sections
        for section in chapter.sections:
            section_state = chapter_state.sections.get(section.id)
            if section_state and section_state.status == SectionStatus.COMPLETED:
                if section_state.generated_content:
                    previous_sections.append((section.title, section_state.generated_content))

        # Process each section sequentially
        for section in chapter.sections:
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
        Generate a single section with retries.
        Returns (success, content).
        """
        # Mark as in progress
        self.state_manager.update_section(state, chapter.id, section.id, status=SectionStatus.IN_PROGRESS)
        self._notify_progress(chapter.id, section.id, "generating")

        # Build prompt
        messages = build_section_prompt(
            section=section,
            chapter=chapter,
            book_title=self.outline.title,
            previous_sections=previous_sections,
        )

        try:
            content = await self.client.generate(messages)

            # Success - save content
            self.state_manager.update_section(
                state,
                chapter.id,
                section.id,
                status=SectionStatus.COMPLETED,
                content=content,
            )

            self._notify_progress(chapter.id, section.id, "completed")
            return True, content

        except OpenRouterError as e:
            # All retries exhausted
            self.state_manager.update_section(
                state,
                chapter.id,
                section.id,
                status=SectionStatus.FAILED,
                error=str(e),
            )

            self._notify_progress(chapter.id, section.id, "failed", str(e))
            return False, None

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

        # Determine filename
        if chapter.id == "preface":
            filename = "00_preface.md"
        elif chapter.id.startswith("appendix_"):
            letter = chapter.id.replace("appendix_", "").upper()
            filename = f"appendix_{letter.lower()}.md"
        else:
            num = int(chapter.id) if chapter.id.isdigit() else 0
            filename = f"chapter_{num:02d}.md"

        filepath = chapters_dir / filename

        # Build chapter content
        lines = []

        # Chapter heading
        if chapter.id == "preface":
            lines.append(f"# Preface: {chapter.title}")
        elif chapter.id.startswith("appendix_"):
            letter = chapter.id.replace("appendix_", "").upper()
            lines.append(f"# Appendix {letter}: {chapter.title}")
        else:
            lines.append(f"# Chapter {chapter.id}: {chapter.title}")

        lines.append("")

        if partial:
            lines.append("> **Note**: This chapter is incomplete due to generation errors.")
            lines.append("")

        # Add each section
        for section in chapter.sections:
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

    # Get all chapter files in order
    chapter_files = []

    # Preface first
    preface_file = chapters_dir / "00_preface.md"
    if preface_file.exists():
        chapter_files.append(preface_file)

    # Main chapters
    for i in range(1, 100):
        chapter_file = chapters_dir / f"chapter_{i:02d}.md"
        if chapter_file.exists():
            chapter_files.append(chapter_file)

    # Appendices
    for letter in "abcdefghijklmnopqrstuvwxyz":
        appendix_file = chapters_dir / f"appendix_{letter}.md"
        if appendix_file.exists():
            chapter_files.append(appendix_file)

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
