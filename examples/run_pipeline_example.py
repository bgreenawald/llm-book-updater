"""
Example script demonstrating how to use the new pipeline system.
"""

import sys
from pathlib import Path
from typing import List

# Add project root to path to allow importing from src
sys.path.append(str(Path(__file__).parent.parent))

from src.config import PhaseConfig, PhaseType, RunConfig
from src.llm_model import GEMINI_PRO, OPENAI_04_MINI
from src.pipeline import run_pipeline


def main():
    # Define the sequence of phases for this run
    # This list can be customized to change the order, repeat phases, or disable them.
    run_phases: List[PhaseConfig] = [
        PhaseConfig(
            phase_type=PhaseType.MODERNIZE,
            model_type=OPENAI_04_MINI,
            temperature=0.3,  # Example: Set a custom temperature
        ),
        PhaseConfig(
            phase_type=PhaseType.EDIT,
            enabled=True,  # Enable the edit phase
        ),
        PhaseConfig(
            phase_type=PhaseType.ANNOTATE,
            model_type=GEMINI_PRO,
        ),
        PhaseConfig(
            phase_type=PhaseType.FINAL,
            model_type=OPENAI_04_MINI,
        ),
        # Example: Add annotation phases
        PhaseConfig(
            phase_type=PhaseType.INTRODUCTION,
            model_type=GEMINI_PRO,
            temperature=0.4,
        ),
        PhaseConfig(
            phase_type=PhaseType.SUMMARY,
            model_type=OPENAI_04_MINI,
            temperature=0.3,
        ),
    ]

    # Create a run configuration from the defined phases
    config = RunConfig(
        book_name="On Liberty",
        author_name="John Stuart Mill",
        input_file=Path("books/On Liberty/markdown/Mill, On Liberty/Mill, On Liberty Clean.md"),
        output_dir=Path("books/On Liberty/markdown/Mill, On Liberty"),
        original_file=Path("books/On Liberty/markdown/Mill, On Liberty/Mill, On Liberty Clean.md"),
        phases=run_phases,
        length_reduction=(30, 50),  # Set length reduction for the entire run
    )

    # Run the pipeline
    run_pipeline(config)


if __name__ == "__main__":
    main()
