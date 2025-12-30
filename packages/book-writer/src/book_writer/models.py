"""Data models for the book writer application."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SectionStatus(str, Enum):
    """Status of a section's generation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ChapterStatus(str, Enum):
    """Status of a chapter's generation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"  # Some sections completed, some failed
    FAILED = "failed"


class SectionOutline(BaseModel):
    """Parsed section from rubric.md."""

    id: str  # e.g., "1.1", "1.2"
    title: str  # e.g., "Core Idea: The Firm as Coordination Mechanism"
    heading_level: int = 2  # Default to H2
    outline_content: str  # H3+ content as the detailed outline
    line_start: int = 0  # Line number in rubric for debugging
    line_end: int = 0


class SectionState(BaseModel):
    """State tracking for a single section."""

    section_id: str
    status: SectionStatus = SectionStatus.PENDING
    retry_count: int = 0
    last_error: Optional[str] = None
    generated_content: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    token_count: Optional[int] = None


class ChapterOutline(BaseModel):
    """Parsed chapter from rubric.md."""

    id: str  # e.g., "1", "2", "preface", "appendix_a"
    number: Optional[int] = None  # Numeric chapter number if applicable
    title: str  # e.g., "What a Firm Is (and Why It Exists)"
    goals: Optional[str] = None  # Chapter Goals section if present
    sections: list[SectionOutline] = Field(default_factory=list)
    summary_box: Optional[str] = None  # Summary box template if present
    line_start: int = 0
    line_end: int = 0


class ChapterState(BaseModel):
    """State tracking for a single chapter."""

    chapter_id: str
    status: ChapterStatus = ChapterStatus.PENDING
    sections: dict[str, SectionState] = Field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class BookOutline(BaseModel):
    """Complete parsed book structure."""

    title: str
    preface: Optional[ChapterOutline] = None
    parts: list[str] = Field(default_factory=list)  # Part titles
    chapters: list[ChapterOutline] = Field(default_factory=list)
    appendices: list[ChapterOutline] = Field(default_factory=list)
    final_notes: Optional[str] = None  # Drafter guidelines


class BookState(BaseModel):
    """Complete state for resume capability."""

    rubric_hash: str  # SHA256 of rubric.md for change detection
    model: str  # OpenRouter model used
    created_at: datetime
    updated_at: datetime
    chapters: dict[str, ChapterState] = Field(default_factory=dict)

    def get_pending_sections(self) -> list[tuple[str, str]]:
        """Return list of (chapter_id, section_id) pairs needing work."""
        pending = []
        for ch_id, ch_state in self.chapters.items():
            for sec_id, sec_state in ch_state.sections.items():
                if sec_state.status in (SectionStatus.PENDING, SectionStatus.FAILED):
                    pending.append((ch_id, sec_id))
        return pending

    def get_completed_sections(self, chapter_id: str) -> list[tuple[str, str]]:
        """Return list of (section_id, content) pairs for completed sections in a chapter."""
        completed = []
        if chapter_id in self.chapters:
            for sec_id, sec_state in self.chapters[chapter_id].sections.items():
                if sec_state.status == SectionStatus.COMPLETED and sec_state.generated_content:
                    completed.append((sec_id, sec_state.generated_content))
        return completed


class BookConfig(BaseModel):
    """Per-book configuration (config.yaml)."""

    title: str = "Untitled Book"
    model: str = "anthropic/claude-sonnet-4"
    max_concurrent_chapters: int = 5


class GenerationConfig(BaseModel):
    """Runtime configuration for generation."""

    model: str = "anthropic/claude-sonnet-4"
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay for exponential backoff
    max_delay: float = 60.0  # Maximum delay cap
    max_concurrent_chapters: int = 5
