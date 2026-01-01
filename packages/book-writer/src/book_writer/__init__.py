"""Book Writer - Generate book drafts from outlines using LLMs."""

__version__ = "0.1.0"

from book_writer.generator import BookGenerator
from book_writer.models import (
    BookOutline,
    BookState,
    ChapterOutline,
    ChapterState,
    ChapterStatus,
    GenerationConfig,
    PhaseModels,
    SectionOutline,
    SectionState,
    SectionStatus,
)
from book_writer.parser import parse_rubric
from book_writer.state import StateManager

__all__ = [
    "__version__",
    # Generator
    "BookGenerator",
    # Models
    "BookOutline",
    "BookState",
    "ChapterOutline",
    "ChapterState",
    "ChapterStatus",
    "GenerationConfig",
    "PhaseModels",
    "SectionOutline",
    "SectionState",
    "SectionStatus",
    # Parser
    "parse_rubric",
    # State
    "StateManager",
]
