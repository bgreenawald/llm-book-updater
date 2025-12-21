from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import Any, Union

from pydantic import Field, ValidationInfo, field_validator, model_validator

from src.common.provider import Provider
from src.constants import (
    DEFAULT_GENERATION_MAX_RETRIES,
    DEFAULT_MAX_SUBBLOCK_TOKENS,
    DEFAULT_MIN_SUBBLOCK_TOKENS,
    DEFAULT_TAGS_TO_PRESERVE,
    MAX_SUBBLOCK_TOKEN_BOUND,
    MIN_SUBBLOCK_TOKEN_BOUND,
)
from src.llm_model import ModelConfig
from src.post_processors import PostProcessor
from src.pydantic_config import BaseConfig


class PhaseType(Enum):
    """Enumeration of available processing phases."""

    MODERNIZE = auto()
    EDIT = auto()
    ANNOTATE = auto()
    FINAL = auto()
    FINAL_TWO_STAGE = auto()  # Two-stage FINAL with identify + implement
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


class TwoStageModelConfig(BaseConfig):
    """Configuration for two-stage phases requiring different models per stage."""

    identify_model: ModelConfig
    implement_model: ModelConfig
    identify_reasoning: dict[str, str] | None = None

    @field_validator("identify_reasoning")
    @classmethod
    def validate_identify_reasoning(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return value
        if not isinstance(value, dict):
            raise ValueError(f"identify_reasoning must be a dict[str, str] or None, got {type(value).__name__}")
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise ValueError(
                    "identify_reasoning must be a dict[str, str]; "
                    f"found key/value types ({type(key).__name__}, {type(item).__name__})"
                )
        return value


class PhaseConfig(BaseConfig):
    """Configuration for a single phase in the pipeline."""

    phase_type: PhaseType
    enabled: bool = True
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(provider=Provider.GEMINI, model_id="google/gemini-2.5-flash")
    )
    reasoning: dict[str, str] | None = None
    llm_kwargs: dict[str, Any] | None = None
    system_prompt_path: Path | None = None
    user_prompt_path: Path | None = None
    custom_output_path: Path | None = None
    # Additional parameters for PhaseFactory integration
    name: str | None = None
    input_file_path: Path | None = None
    output_file_path: Path | None = None
    original_file_path: Path | None = None
    llm_model_instance: Any | None = None
    # Unified post-processors list that can contain strings (built-in) or
    # PostProcessor instances (custom)
    post_processors: list[Union[str, PostProcessor, PostProcessorType]] | None = None
    book_name: str | None = None
    author_name: str | None = None
    # Batch processing parameters
    use_batch: bool = False
    batch_size: int | None = Field(default=None, gt=0)
    # Retry configuration for failed generations
    # When False (default), any generation failure immediately stops the pipeline
    # When True, failed generations are retried up to max_retries times
    enable_retry: bool = False
    max_retries: int = Field(default=DEFAULT_GENERATION_MAX_RETRIES, ge=0)
    # Sub-block processing parameters
    # When enabled, large chapter bodies are split into smaller sub-blocks for processing
    use_subblocks: bool = False
    max_subblock_tokens: int = Field(
        default=DEFAULT_MAX_SUBBLOCK_TOKENS,
        ge=MIN_SUBBLOCK_TOKEN_BOUND,
        le=MAX_SUBBLOCK_TOKEN_BOUND,
    )
    min_subblock_tokens: int = Field(
        default=DEFAULT_MIN_SUBBLOCK_TOKENS,
        ge=MIN_SUBBLOCK_TOKEN_BOUND,
        le=MAX_SUBBLOCK_TOKEN_BOUND,
    )
    # Block skipping parameter
    # If set, blocks with fewer tokens than this value will be skipped entirely
    # (before chunking into subblocks). Default None means no skipping.
    skip_if_less_than_tokens: int | None = None
    # Two-stage phase configuration (required for FINAL_TWO_STAGE)
    two_stage_config: TwoStageModelConfig | None = None

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return value
        if not isinstance(value, dict):
            raise ValueError(f"reasoning must be a dict[str, str] or None, got {type(value).__name__}")
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise ValueError(
                    "reasoning must be a dict[str, str]; "
                    f"found key/value types ({type(key).__name__}, {type(item).__name__})"
                )
        return value

    @field_validator("llm_kwargs")
    @classmethod
    def validate_llm_kwargs(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return value
        if not isinstance(value, dict):
            raise ValueError(f"llm_kwargs must be a dict or None, got {type(value).__name__}")
        return value

    @field_validator("use_subblocks", mode="before")
    @classmethod
    def validate_use_subblocks_type(cls, value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be a bool")
        return value

    @field_validator("max_subblock_tokens", "min_subblock_tokens", mode="before")
    @classmethod
    def validate_subblock_token_types(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{info.field_name} must be an int")
        return value

    @model_validator(mode="after")
    def validate_phase_config(self) -> "PhaseConfig":
        if self.batch_size is not None and not self.use_batch:
            raise ValueError(
                "batch_size was set but use_batch is False; either enable use_batch or set batch_size=None"
            )

        if self.max_subblock_tokens <= self.min_subblock_tokens:
            raise ValueError(
                "max_subblock_tokens must be greater than min_subblock_tokens, "
                f"got max={self.max_subblock_tokens}, min={self.min_subblock_tokens}"
            )

        if self.phase_type == PhaseType.FINAL_TWO_STAGE:
            if self.two_stage_config is None:
                raise ValueError("two_stage_config is required for FINAL_TWO_STAGE phase")
        elif self.two_stage_config is not None:
            raise ValueError(f"two_stage_config is only valid for FINAL_TWO_STAGE phase, not {self.phase_type.name}")

        if self.phase_type != PhaseType.FINAL_TWO_STAGE:
            if self.system_prompt_path is None:
                self.system_prompt_path = Path(f"./prompts/{self.phase_type.name.lower()}_system.md")
            if self.user_prompt_path is None:
                self.user_prompt_path = Path(f"./prompts/{self.phase_type.name.lower()}_user.md")

        if self.name is None:
            self.name = self.phase_type.name.lower()

        return self


class RunConfig(BaseConfig):
    """Configuration for a complete run of the pipeline."""

    book_id: str
    book_name: str
    author_name: str
    input_file: Path
    output_dir: Path
    original_file: Path
    phases: list[PhaseConfig] = Field(default_factory=list)
    # Tags to preserve during processing (f-string tags like {preface}, {license})
    tags_to_preserve: list[str] = Field(default_factory=lambda: list(DEFAULT_TAGS_TO_PRESERVE))
    # Maximum number of workers for parallel processing across all phases
    max_workers: int | None = Field(default=None, gt=0)
    # Phase index to start execution from (0-based). Phases before this will be skipped.
    # Useful for resuming after a failed phase.
    start_from_phase: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_run_config(self) -> "RunConfig":
        if not self.phases and self.start_from_phase != 0:
            raise ValueError(
                f"start_from_phase is out of range for {len(self.phases)} phases: got {self.start_from_phase}"
            )
        if self.phases and self.start_from_phase >= len(self.phases):
            raise ValueError(
                f"start_from_phase is out of range for {len(self.phases)} phases: got {self.start_from_phase}"
            )

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"output_dir is not a directory: {exc}") from exc

        return self

    def get_phase_order(self) -> list[PhaseType]:
        """Get the ordered list of phase types."""
        return [phase.phase_type for phase in self.phases]
