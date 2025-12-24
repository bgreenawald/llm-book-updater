"""
Example script demonstrating the new pipeline system with phase factory.
"""

import sys
from pathlib import Path
from typing import List

# Add project root to path to allow importing from src
sys.path.append(str(Path(__file__).parent.parent))

from src.api.config import PhaseConfig, PhaseType, RunConfig
from src.core.pipeline import run_pipeline
from src.models import GEMINI_PRO, OPENAI_04_MINI


def main() -> None:
    """Run the pipeline with different phase types."""

    # Define phases with different types
    run_phases: List[PhaseConfig] = [
        # Standard phases (use StandardLlmPhase)
        PhaseConfig(
            phase_type=PhaseType.MODERNIZE,
            model=OPENAI_04_MINI,
            # Note: Temperature can be passed via llm_kwargs if needed: llm_kwargs={"temperature": 0.3}
        ),
        PhaseConfig(
            phase_type=PhaseType.EDIT,
            enabled=False,  # Disable this phase
        ),
        PhaseConfig(
            phase_type=PhaseType.ANNOTATE,
            model=GEMINI_PRO,
        ),
        PhaseConfig(
            phase_type=PhaseType.FINAL,
            model=OPENAI_04_MINI,
        ),
        # Annotation phases (use specific annotation classes)
        PhaseConfig(
            phase_type=PhaseType.INTRODUCTION,
            model=GEMINI_PRO,
        ),
        PhaseConfig(
            phase_type=PhaseType.SUMMARY,
            model=OPENAI_04_MINI,
        ),
    ]

    # Create run configuration
    config = RunConfig(
        book_id="on_liberty",
        book_name="On Liberty",
        author_name="John Stuart Mill",
        input_file=Path("books/On Liberty/markdown/Mill, On Liberty/Mill, On Liberty Clean.md"),
        output_dir=Path("books/On Liberty/markdown/Mill, On Liberty"),
        original_file=Path("books/On Liberty/markdown/Mill, On Liberty/Mill, On Liberty Clean.md"),
        phases=run_phases,
    )

    # Run the pipeline
    run_pipeline(config)


if __name__ == "__main__":
    main()
