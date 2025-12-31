"""
Example script demonstrating how to use the new pipeline system.
"""

from pathlib import Path
from typing import List

from book_updater import PhaseConfig, PhaseType, RunConfig
from book_updater.pipeline import run_pipeline
from llm_core import GEMINI_PRO, OPENAI_04_MINI


def main() -> None:
    # Define the sequence of phases for this run
    # This list can be customized to change the order, repeat phases, or disable them.
    run_phases: List[PhaseConfig] = [
        PhaseConfig(
            phase_type=PhaseType.MODERNIZE,
            model=OPENAI_04_MINI,
            # Note: Temperature can be passed via llm_kwargs if needed: llm_kwargs={"temperature": 0.3}
        ),
        PhaseConfig(
            phase_type=PhaseType.EDIT,
            enabled=True,  # Enable the edit phase
        ),
        PhaseConfig(
            phase_type=PhaseType.ANNOTATE,
            model=GEMINI_PRO,
        ),
        PhaseConfig(
            phase_type=PhaseType.FINAL,
            model=OPENAI_04_MINI,
        ),
        # Example: Add annotation phases
        PhaseConfig(
            phase_type=PhaseType.INTRODUCTION,
            model=GEMINI_PRO,
        ),
        PhaseConfig(
            phase_type=PhaseType.SUMMARY,
            model=OPENAI_04_MINI,
        ),
    ]

    # Create a run configuration from the defined phases
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
