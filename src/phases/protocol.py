"""Protocol definition for phases in the LLM processing pipeline.

This module defines the Phase Protocol that specifies the contract all phase
implementations must satisfy. Using Protocol allows structural subtyping -
any class with these attributes/methods works, no inheritance required.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from src.processing.post_processors import PostProcessorChain


@runtime_checkable
class Phase(Protocol):
    """Protocol defining what Pipeline requires from a phase.

    This is the contract that all phase implementations must satisfy.
    Using Protocol allows structural subtyping - any class with these
    attributes/methods works, no inheritance required.

    Attributes required by Pipeline:
        name: Unique identifier for the phase
        input_file_path: Path to the input markdown file
        output_file_path: Path where processed output will be written
        start_token_count: Approximate token count at start (set during run())
        end_token_count: Approximate token count at end (set during run())
        system_prompt_path: Path to system prompt file (can be None)
        user_prompt_path: Path to user prompt file (can be None)
        system_prompt: Fully rendered system prompt text (for metadata)
        post_processor_chain: Optional chain of post-processors
        llm_kwargs: Additional kwargs to pass to LLM calls (e.g., provider parameters)
    """

    # Core identification
    name: str

    # File paths
    input_file_path: Path
    output_file_path: Path

    # Token tracking (set by run())
    start_token_count: Optional[int]
    end_token_count: Optional[int]

    # For metadata collection (can be None for phases that don't use them)
    system_prompt_path: Optional[Path]
    user_prompt_path: Optional[Path]
    system_prompt: str  # The rendered system prompt (for metadata)

    # Post-processing
    post_processor_chain: Optional[PostProcessorChain]

    # LLM call parameters
    llm_kwargs: Dict[str, Any]

    def run(self, **kwargs) -> None:
        """Execute the phase processing.

        Args:
            **kwargs: Additional arguments passed from Pipeline
        """
        ...
