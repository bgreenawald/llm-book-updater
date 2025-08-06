from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from src.llm_model import GEMINI_FLASH

if TYPE_CHECKING:
    from src.llm_model import LlmModel
    from src.post_processors import PostProcessor


class PhaseType(Enum):
    """Enumeration of available processing phases."""

    MODERNIZE = auto()
    EDIT = auto()
    ANNOTATE = auto()
    FINAL = auto()
    INTRODUCTION = auto()
    SUMMARY = auto()


class PostProcessorType(Enum):
    """Enumeration of available post-processor types."""

    # Basic formatting processors
    ENSURE_BLANK_LINE = auto()
    REMOVE_TRAILING_WHITESPACE = auto()
    REMOVE_XML_TAGS = auto()
    REMOVE_BLANK_LINES_IN_LIST = auto()

    # Content preservation processors
    NO_NEW_HEADERS = auto()
    REVERT_REMOVED_BLOCK_LINES = auto()
    PRESERVE_F_STRING_TAGS = auto()

    # Specialized processors
    ORDER_QUOTE_ANNOTATION = auto()


@dataclass
class PhaseConfig:
    """Configuration for a single phase in the pipeline."""

    phase_type: PhaseType
    enabled: bool = True
    model_type: str = GEMINI_FLASH
    temperature: float = 0.2
    reasoning: Optional[Dict[str, Dict[str, str]]] = None
    system_prompt_path: Optional[Path] = None
    user_prompt_path: Optional[Path] = None
    custom_output_path: Optional[Path] = None
    # Additional parameters for PhaseFactory integration
    name: Optional[str] = None
    input_file_path: Optional[Path] = None
    output_file_path: Optional[Path] = None
    original_file_path: Optional[Path] = None
    model: Optional["LlmModel"] = None  # LlmModel instance
    # Unified post-processors list that can contain strings (built-in) or
    # PostProcessor instances (custom)
    post_processors: Optional[List[Union[str, "PostProcessor", PostProcessorType]]] = None
    book_name: Optional[str] = None
    author_name: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Post-initialization to set default prompt paths and name if not provided.
        """
        if self.system_prompt_path is None:
            self.system_prompt_path = Path(f"./prompts/{self.phase_type.name.lower()}_system.md")
        if self.user_prompt_path is None:
            self.user_prompt_path = Path(f"./prompts/{self.phase_type.name.lower()}_user.md")
        if self.name is None:
            self.name = self.phase_type.name.lower()


@dataclass
class RunConfig:
    """Configuration for a complete run of the pipeline."""

    book_id: str
    book_name: str
    author_name: str
    input_file: Path
    output_dir: Path
    original_file: Path
    phases: List[PhaseConfig] = field(default_factory=list)
    # Length reduction parameter for the entire run (can be int or tuple of bounds)
    # Default to 35-50% reduction if not specified
    length_reduction: Optional[Union[int, Tuple[int, int]]] = (35, 50)
    # Tags to preserve during processing (f-string tags like {preface}, {license})
    tags_to_preserve: List[str] = field(default_factory=lambda: ["{preface}", "{license}"])
    # Maximum number of workers for parallel processing across all phases
    max_workers: Optional[int] = None

    def __str__(self) -> str:
        return (
            f"RunConfig(book_id={self.book_id}, "
            f"book_name={self.book_name}, "
            f"author_name={self.author_name}, "
            f"input_file={self.input_file}, "
            f"output_dir={self.output_dir}, "
            f"original_file={self.original_file}, "
            f"phases={self.phases}, "
            f"length_reduction={self.length_reduction}, "
            f"tags_to_preserve={self.tags_to_preserve})"
        )

    def __repr__(self) -> str:
        return (
            f"RunConfig(book_id={self.book_id}, "
            f"book_name={self.book_name}, "
            f"author_name={self.author_name}, "
            f"input_file={self.input_file}, "
            f"output_dir={self.output_dir}, "
            f"original_file={self.original_file}, "
            f"phases={self.phases}, "
            f"length_reduction={self.length_reduction}, "
            f"tags_to_preserve={self.tags_to_preserve})"
        )

    def __post_init__(self) -> None:
        """
        Post-initialization to ensure the output directory exists.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_phase_order(self) -> List[PhaseType]:
        """Get the ordered list of phase types."""
        return [phase.phase_type for phase in self.phases]
