from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

from src.llm_model import GEMINI_FLASH


class PhaseType(Enum):
    MODERNIZE = auto()
    EDIT = auto()
    ANNOTATE = auto()
    FINAL = auto()
    FORMATTING = auto()


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
    max_workers: Optional[int] = None

    def __post_init__(self):
        if self.system_prompt_path is None:
            self.system_prompt_path = Path(
                f"./prompts/{self.phase_type.name.lower()}.md"
            )


@dataclass
class RunConfig:
    """Configuration for a complete run of the pipeline."""

    book_name: str
    author_name: str
    input_file: Path
    output_dir: Path
    original_file: Path
    phases: List[PhaseConfig] = field(default_factory=list)

    def __str__(self):
        return f"RunConfig(book_name={self.book_name}, author_name={self.author_name}, input_file={self.input_file}, output_dir={self.output_dir}, original_file={self.original_file}, phases={self.phases})"

    def __repr__(self):
        return f"RunConfig(book_name={self.book_name}, author_name={self.author_name}, input_file={self.input_file}, output_dir={self.output_dir}, original_file={self.original_file}, phases={self.phases})"

    def __post_init__(self):
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_phase_order(self) -> List[PhaseType]:
        """Get the ordered list of phase types."""
        return [phase.phase_type for phase in self.phases]
