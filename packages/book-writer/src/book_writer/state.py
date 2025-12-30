"""State management for book generation with resume capability."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import (
    BookOutline,
    BookState,
    ChapterState,
    ChapterStatus,
    SectionState,
    SectionStatus,
)


class StateManager:
    """Manages persistent state for book generation."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.state_file = output_dir / "state.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> Optional[BookState]:
        """Load existing state from disk, return None if not found."""
        if not self.state_file.exists():
            return None

        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            return BookState.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            # Log error and return None to trigger reinitialization
            print(f"Warning: Could not load state file: {e}")
            return None

    def save_state(self, state: BookState) -> None:
        """Atomically save state to disk (write to temp, rename)."""
        state.updated_at = datetime.now()

        # Write to temp file first, then rename for atomicity
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.output_dir,
            delete=False,
            suffix=".json",
            encoding="utf-8",
        ) as f:
            f.write(state.model_dump_json(indent=2))
            temp_path = Path(f.name)

        # Atomic rename
        temp_path.rename(self.state_file)

    def initialize_state(self, outline: BookOutline, model: str, rubric_hash: str) -> BookState:
        """Create fresh state from book outline."""
        now = datetime.now()

        chapters: dict[str, ChapterState] = {}

        # Add preface if present
        if outline.preface:
            chapters[outline.preface.id] = self._create_chapter_state(outline.preface)

        # Add all chapters
        for chapter in outline.chapters:
            chapters[chapter.id] = self._create_chapter_state(chapter)

        # Add appendices
        for appendix in outline.appendices:
            chapters[appendix.id] = self._create_chapter_state(appendix)

        state = BookState(
            rubric_hash=rubric_hash,
            model=model,
            created_at=now,
            updated_at=now,
            chapters=chapters,
        )

        self.save_state(state)
        return state

    def _create_chapter_state(self, chapter) -> ChapterState:
        """Create initial state for a chapter."""
        sections = {}
        for section in chapter.sections:
            sections[section.id] = SectionState(section_id=section.id)

        return ChapterState(chapter_id=chapter.id, sections=sections)

    def update_section(
        self,
        state: BookState,
        chapter_id: str,
        section_id: str,
        status: SectionStatus,
        content: Optional[str] = None,
        error: Optional[str] = None,
        token_count: Optional[int] = None,
    ) -> BookState:
        """Update section state and persist immediately."""
        if chapter_id not in state.chapters:
            raise ValueError(f"Chapter {chapter_id} not found in state")

        chapter_state = state.chapters[chapter_id]
        if section_id not in chapter_state.sections:
            raise ValueError(f"Section {section_id} not found in chapter {chapter_id}")

        section_state = chapter_state.sections[section_id]

        # Update section state
        section_state.status = status

        if status == SectionStatus.IN_PROGRESS:
            section_state.started_at = datetime.now()
        elif status == SectionStatus.COMPLETED:
            section_state.completed_at = datetime.now()
            section_state.generated_content = content
            section_state.token_count = token_count
        elif status == SectionStatus.FAILED:
            section_state.last_error = error
            section_state.retry_count += 1

        # Update chapter status
        self._update_chapter_status(chapter_state)

        # Persist immediately
        self.save_state(state)
        return state

    def _update_chapter_status(self, chapter_state: ChapterState) -> None:
        """Recalculate chapter status based on section states."""
        statuses = [s.status for s in chapter_state.sections.values()]

        if not statuses:
            chapter_state.status = ChapterStatus.COMPLETED
            return

        if all(s == SectionStatus.COMPLETED for s in statuses):
            chapter_state.status = ChapterStatus.COMPLETED
            chapter_state.completed_at = datetime.now()
        elif all(s == SectionStatus.PENDING for s in statuses):
            chapter_state.status = ChapterStatus.PENDING
        elif any(s == SectionStatus.FAILED for s in statuses):
            # Check if any completed
            if any(s == SectionStatus.COMPLETED for s in statuses):
                chapter_state.status = ChapterStatus.PARTIAL
            else:
                chapter_state.status = ChapterStatus.FAILED
        elif any(s == SectionStatus.IN_PROGRESS for s in statuses):
            chapter_state.status = ChapterStatus.IN_PROGRESS
        else:
            # Mix of completed and pending
            chapter_state.status = ChapterStatus.IN_PROGRESS

    def mark_chapter_started(self, state: BookState, chapter_id: str) -> BookState:
        """Mark a chapter as started."""
        if chapter_id in state.chapters:
            state.chapters[chapter_id].status = ChapterStatus.IN_PROGRESS
            state.chapters[chapter_id].started_at = datetime.now()
            self.save_state(state)
        return state

    def should_reinitialize(self, state: BookState, rubric_hash: str) -> bool:
        """Check if rubric changed, requiring new state."""
        return state.rubric_hash != rubric_hash

    def reset_failed_sections(self, state: BookState) -> BookState:
        """Reset all failed sections to pending for retry."""
        for chapter_state in state.chapters.values():
            for section_state in chapter_state.sections.values():
                if section_state.status == SectionStatus.FAILED:
                    section_state.status = SectionStatus.PENDING
                    section_state.retry_count = 0
                    section_state.last_error = None

            self._update_chapter_status(chapter_state)

        self.save_state(state)
        return state

    def get_chapter_progress(self, state: BookState, chapter_id: str) -> dict:
        """Get progress summary for a chapter."""
        if chapter_id not in state.chapters:
            return {"total": 0, "completed": 0, "failed": 0, "pending": 0}

        chapter_state = state.chapters[chapter_id]
        statuses = [s.status for s in chapter_state.sections.values()]

        return {
            "total": len(statuses),
            "completed": sum(1 for s in statuses if s == SectionStatus.COMPLETED),
            "failed": sum(1 for s in statuses if s == SectionStatus.FAILED),
            "pending": sum(1 for s in statuses if s == SectionStatus.PENDING),
            "in_progress": sum(1 for s in statuses if s == SectionStatus.IN_PROGRESS),
        }

    def get_overall_progress(self, state: BookState) -> dict:
        """Get overall progress summary."""
        total_sections = 0
        completed = 0
        failed = 0
        pending = 0

        for chapter_state in state.chapters.values():
            for section_state in chapter_state.sections.values():
                total_sections += 1
                if section_state.status == SectionStatus.COMPLETED:
                    completed += 1
                elif section_state.status == SectionStatus.FAILED:
                    failed += 1
                elif section_state.status == SectionStatus.PENDING:
                    pending += 1

        return {
            "total_chapters": len(state.chapters),
            "total_sections": total_sections,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "in_progress": total_sections - completed - failed - pending,
        }
