from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from src.common.provider import Provider
from src.constants import DEFAULT_LENGTH_REDUCTION_BOUNDS, DEFAULT_TAGS_TO_PRESERVE
from src.llm_model import ModelConfig

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
    REMOVE_MARKDOWN_BLOCKS = auto()

    # Content preservation processors
    NO_NEW_HEADERS = auto()
    REVERT_REMOVED_BLOCK_LINES = auto()
    PRESERVE_F_STRING_TAGS = auto()

    # Specialized processors
    ORDER_QUOTE_ANNOTATION = auto()

    # Validation processors
    VALIDATE_NON_EMPTY_SECTION = auto()


def _validate_temperature(*, temperature: float) -> None:
    """Validate temperature configuration.

    Args:
        temperature: Sampling temperature. Must be between 0 and 2 (inclusive).

    Raises:
        TypeError: If temperature is not a number.
        ValueError: If temperature is out of range.
    """
    if not isinstance(temperature, (int, float)):
        raise TypeError(f"Temperature must be a number, got {type(temperature).__name__}")
    if temperature < 0 or temperature > 2:
        raise ValueError(f"Temperature must be between 0 and 2, got {temperature}")


def _validate_length_reduction(*, length_reduction: Optional[Union[int, Tuple[int, int]]]) -> None:
    """Validate length reduction configuration.

    Length reduction is represented as a percentage, used for prompt formatting
    (e.g., ``35%`` or ``35-50%``).

    Args:
        length_reduction: Either a single percentage (int) or a 2-tuple of bounds.

    Raises:
        TypeError: If the input type is invalid.
        ValueError: If the value is out of range or bounds are malformed.
    """
    if length_reduction is None:
        return

    if isinstance(length_reduction, int):
        if length_reduction < 0 or length_reduction > 100:
            raise ValueError(f"length_reduction must be between 0 and 100, got {length_reduction}")
        return

    if isinstance(length_reduction, tuple):
        if len(length_reduction) != 2:
            raise ValueError(f"length_reduction tuple must have exactly 2 values, got {len(length_reduction)}")
        low, high = length_reduction
        if not isinstance(low, int) or not isinstance(high, int):
            raise TypeError(
                f"length_reduction tuple values must both be ints, got ({type(low).__name__}, {type(high).__name__})"
            )
        if low < 0 or high < 0 or low > 100 or high > 100:
            raise ValueError(f"length_reduction bounds must be between 0 and 100, got {length_reduction}")
        if low > high:
            raise ValueError(f"length_reduction lower bound must be <= upper bound, got {length_reduction}")
        return

    raise TypeError(
        "length_reduction must be an int percentage, a 2-tuple of int bounds, or None; "
        f"got {type(length_reduction).__name__}"
    )


@dataclass
class PhaseConfig:
    """Configuration for a single phase in the pipeline."""

    phase_type: PhaseType
    enabled: bool = True
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(Provider.GEMINI, "google/gemini-2.5-flash", "gemini-2.5-flash")
    )
    temperature: float = 0.2
    reasoning: Optional[dict[str, str]] = None
    system_prompt_path: Optional[Path] = None
    user_prompt_path: Optional[Path] = None
    custom_output_path: Optional[Path] = None
    # Additional parameters for PhaseFactory integration
    name: Optional[str] = None
    input_file_path: Optional[Path] = None
    output_file_path: Optional[Path] = None
    original_file_path: Optional[Path] = None
    llm_model_instance: Optional["LlmModel"] = None  # LlmModel instance
    # Unified post-processors list that can contain strings (built-in) or
    # PostProcessor instances (custom)
    post_processors: Optional[List[Union[str, "PostProcessor", PostProcessorType]]] = None
    book_name: Optional[str] = None
    author_name: Optional[str] = None
    # Batch processing parameters
    use_batch: bool = False
    batch_size: Optional[int] = None

    def __post_init__(self) -> None:
        """
        Post-initialization to validate configuration and set defaults.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If a field has an invalid value.
        """
        if not isinstance(self.phase_type, PhaseType):
            raise TypeError(f"phase_type must be a PhaseType, got {type(self.phase_type).__name__}")

        _validate_temperature(temperature=self.temperature)

        if self.reasoning is not None:
            if not isinstance(self.reasoning, dict):
                raise TypeError(f"reasoning must be a dict[str, str] or None, got {type(self.reasoning).__name__}")
            for k, v in self.reasoning.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise TypeError(
                        "reasoning must be a dict[str, str]; "
                        f"found key/value types ({type(k).__name__}, {type(v).__name__})"
                    )

        if not isinstance(self.use_batch, bool):
            raise TypeError(f"use_batch must be a bool, got {type(self.use_batch).__name__}")

        if self.batch_size is not None:
            if not isinstance(self.batch_size, int):
                raise TypeError(f"batch_size must be an int or None, got {type(self.batch_size).__name__}")
            if self.batch_size <= 0:
                raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
            if not self.use_batch:
                raise ValueError(
                    "batch_size was set but use_batch is False; either enable use_batch or set batch_size=None"
                )

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
    length_reduction: Optional[Union[int, Tuple[int, int]]] = DEFAULT_LENGTH_REDUCTION_BOUNDS
    # Tags to preserve during processing (f-string tags like {preface}, {license})
    tags_to_preserve: List[str] = field(default_factory=lambda: list(DEFAULT_TAGS_TO_PRESERVE))
    # Maximum number of workers for parallel processing across all phases
    max_workers: Optional[int] = None
    # Phase index to start execution from (0-based). Phases before this will be skipped.
    # Useful for resuming after a failed phase.
    start_from_phase: int = 0

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
        Post-initialization to validate configuration and ensure output directory exists.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If a field has an invalid value.
        """
        _validate_length_reduction(length_reduction=self.length_reduction)

        if self.max_workers is not None:
            if not isinstance(self.max_workers, int):
                raise TypeError(f"max_workers must be an int or None, got {type(self.max_workers).__name__}")
            if self.max_workers <= 0:
                raise ValueError(f"max_workers must be > 0, got {self.max_workers}")

        if not isinstance(self.start_from_phase, int):
            raise TypeError(f"start_from_phase must be an int, got {type(self.start_from_phase).__name__}")
        if self.start_from_phase < 0:
            raise ValueError(f"start_from_phase must be >= 0, got {self.start_from_phase}")
        if len(self.phases) == 0:
            if self.start_from_phase != 0:
                raise ValueError(
                    f"start_from_phase is out of range for {len(self.phases)} phases: got {self.start_from_phase}"
                )
        elif self.start_from_phase >= len(self.phases):
            raise ValueError(
                f"start_from_phase is out of range for {len(self.phases)} phases: got {self.start_from_phase}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_phase_order(self) -> List[PhaseType]:
        """Get the ordered list of phase types."""
        return [phase.phase_type for phase in self.phases]
