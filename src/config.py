from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

from src.llm_model import ModelType


class PhaseType(Enum):
    MODERNIZE = auto()
    EDIT = auto()
    ANNOTATE = auto()
    FINAL = auto()


@dataclass
class PhaseConfig:
    """Configuration for a single phase in the pipeline."""

    phase_type: PhaseType
    enabled: bool = True
    model_type: ModelType = ModelType.GEMINI_FLASH
    temperature: float = 0.2
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
    phases: Dict[PhaseType, PhaseConfig] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set default phases if none provided
        if not self.phases:
            self.phases = {
                PhaseType.MODERNIZE: PhaseConfig(
                    phase_type=PhaseType.MODERNIZE,
                    model_type=ModelType.GEMINI_FLASH,
                    temperature=0.2,
                ),
                PhaseType.EDIT: PhaseConfig(
                    phase_type=PhaseType.EDIT,
                    model_type=ModelType.GEMINI_PRO,
                    temperature=0.2,
                ),
                PhaseType.ANNOTATE: PhaseConfig(
                    phase_type=PhaseType.ANNOTATE,
                    model_type=ModelType.GEMINI_FLASH,
                    temperature=0.2,
                ),
                PhaseType.FINAL: PhaseConfig(
                    phase_type=PhaseType.FINAL,
                    model_type=ModelType.GEMINI_PRO,
                    temperature=0.2,
                ),
            }

    def get_phase_order(self) -> List[PhaseType]:
        """Get the ordered list of phase types."""
        return [
            PhaseType.MODERNIZE,
            PhaseType.EDIT,
            PhaseType.ANNOTATE,
            PhaseType.FINAL,
        ]

    def get_phase_config(self, phase_type: PhaseType) -> PhaseConfig:
        """Get configuration for a specific phase."""
        return self.phases.get(phase_type)
