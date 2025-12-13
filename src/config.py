from enum import Enum, auto
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

    # Content preservation processors
    NO_NEW_HEADERS = auto()
    REVERT_REMOVED_BLOCK_LINES = auto()
    PRESERVE_F_STRING_TAGS = auto()

    # Specialized processors
    ORDER_QUOTE_ANNOTATION = auto()


def _parse_length_reduction(*, value: Any) -> Optional[Union[int, Tuple[int, int]]]:
    """Parse/validate a length reduction specification.

    Args:
        value: A length reduction input. Supports:
            - None
            - int percent (0-100)
            - 2-tuple/list of ints (0-100) where low <= high
            - str like "35", "35%", "35-50", or "35-50%"

    Returns:
        Parsed value as int, (low, high) tuple, or None.

    Raises:
        TypeError: If the value type is invalid.
        ValueError: If the value is out of range or malformed.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        # bool is a subclass of int; reject it explicitly
        raise TypeError("length_reduction must be an int/tuple/str/None, not bool")

    if isinstance(value, int):
        if value < 0 or value > 100:
            raise ValueError(f"length_reduction must be between 0 and 100, got {value}")
        return value

    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"length_reduction tuple must have exactly 2 values, got {len(value)}")
        low, high = value[0], value[1]
        if not isinstance(low, int) or not isinstance(high, int):
            raise TypeError(
                f"length_reduction tuple values must both be ints, got ({type(low).__name__}, {type(high).__name__})"
            )
        if low < 0 or high < 0 or low > 100 or high > 100:
            raise ValueError(f"length_reduction bounds must be between 0 and 100, got ({low}, {high})")
        if low > high:
            raise ValueError(f"length_reduction lower bound must be <= upper bound, got ({low}, {high})")
        return (low, high)

    if isinstance(value, str):
        s = value.strip().lower()
        s = s.removesuffix("%")
        s = s.replace("–", "-").replace("—", "-")
        s = re.sub(r"\s+", "", s)
        if not s:
            raise ValueError("length_reduction cannot be an empty string")
        if "-" in s:
            low_s, high_s = s.split("-", 1)
            if not low_s or not high_s:
                raise ValueError(f"length_reduction range must be like '35-50', got {value!r}")
            return _parse_length_reduction(value=(int(low_s), int(high_s)))
        return _parse_length_reduction(value=int(s))

    raise TypeError(
        "length_reduction must be an int percentage, a 2-tuple/list of int bounds, a string like '35-50%', or None; "
        f"got {type(value).__name__}"
    )


class PhaseConfig(BaseModel):
    """Configuration for a single phase in the pipeline."""

    phase_type: PhaseType
    enabled: bool = True
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider=Provider.GEMINI,
            model_id="google/gemini-2.5-flash",
            provider_model_name="gemini-2.5-flash",
        )
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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    @field_validator("phase_type", mode="before")
    @classmethod
    def _coerce_phase_type(cls, v: Any) -> Any:
        """Accept a PhaseType or a string phase name like 'modernize'."""
        if isinstance(v, PhaseType):
            return v
        if isinstance(v, str):
            try:
                return PhaseType[v.strip().upper()]
            except KeyError as e:
                raise ValueError(f"Invalid phase_type: {v!r}") from e
        return v

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        """Validate temperature is within [0, 2]."""
        if v < 0 or v > 2:
            raise ValueError(f"temperature must be between 0 and 2, got {v}")
        return v

    @field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasoning(cls, v: Any) -> Any:
        """Coerce reasoning dict keys/values to strings when possible."""
        if v is None:
            return None
        if not isinstance(v, dict):
            raise TypeError(f"reasoning must be a dict[str, str] or None, got {type(v).__name__}")
        return {str(k): str(val) for k, val in v.items()}

    @model_validator(mode="after")
    def _set_defaults_and_validate_batch(self) -> "PhaseConfig":
        """Set default prompt paths/name and validate batch options."""
        if self.batch_size is not None:
            if self.batch_size <= 0:
                raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
            if not self.use_batch:
                raise ValueError("batch_size was set but use_batch is False; either enable use_batch or set batch_size=None")

        if self.system_prompt_path is None:
            self.system_prompt_path = Path(f"./prompts/{self.phase_type.name.lower()}_system.md")
        if self.user_prompt_path is None:
            self.user_prompt_path = Path(f"./prompts/{self.phase_type.name.lower()}_user.md")
        if self.name is None:
            self.name = self.phase_type.name.lower()

        return self


class RunConfig(BaseModel):
    """Configuration for a complete run of the pipeline."""

    book_id: str
    book_name: str
    author_name: str
    input_file: Path
    output_dir: Path
    original_file: Path
    phases: List[PhaseConfig] = Field(default_factory=list)
    # Length reduction parameter for the entire run (can be int or tuple of bounds)
    # Default to 35-50% reduction if not specified
    length_reduction: Optional[Union[int, Tuple[int, int]]] = DEFAULT_LENGTH_REDUCTION_BOUNDS
    # Tags to preserve during processing (f-string tags like {preface}, {license})
    tags_to_preserve: List[str] = Field(default_factory=lambda: list(DEFAULT_TAGS_TO_PRESERVE))
    # Maximum number of workers for parallel processing across all phases
    max_workers: Optional[int] = None
    # Phase index to start execution from (0-based). Phases before this will be skipped.
    # Useful for resuming after a failed phase.
    start_from_phase: int = 0

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

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

    @field_validator("length_reduction", mode="before")
    @classmethod
    def _coerce_length_reduction(cls, v: Any) -> Any:
        """Coerce length_reduction from common formats like '35-50%'."""
        return _parse_length_reduction(value=v)

    @field_validator("tags_to_preserve", mode="before")
    @classmethod
    def _coerce_tags_to_preserve(cls, v: Any) -> Any:
        """Allow tags_to_preserve to be a list/tuple or a comma-separated string."""
        if v is None:
            return list(DEFAULT_TAGS_TO_PRESERVE)
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",")]
            return [p for p in parts if p]
        return v

    @field_validator("max_workers")
    @classmethod
    def _validate_max_workers(cls, v: Optional[int]) -> Optional[int]:
        """Validate max_workers is positive when set."""
        if v is None:
            return None
        if v <= 0:
            raise ValueError(f"max_workers must be > 0, got {v}")
        return v

    @field_validator("start_from_phase")
    @classmethod
    def _validate_start_from_phase(cls, v: int) -> int:
        """Validate start_from_phase is non-negative."""
        if v < 0:
            raise ValueError(f"start_from_phase must be >= 0, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_phase_bounds_and_prepare_output_dir(self) -> "RunConfig":
        """Validate phase bounds and ensure output directory exists."""
        if len(self.phases) == 0:
            if self.start_from_phase != 0:
                raise ValueError(f"start_from_phase is out of range for {len(self.phases)} phases: got {self.start_from_phase}")
        elif self.start_from_phase >= len(self.phases):
            raise ValueError(f"start_from_phase is out of range for {len(self.phases)} phases: got {self.start_from_phase}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def get_phase_order(self) -> List[PhaseType]:
        """Get the ordered list of phase types."""
        return [phase.phase_type for phase in self.phases]
