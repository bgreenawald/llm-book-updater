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
            enabled=False,  # Example: Disable the edit phase
        ),
        PhaseConfig(
            phase_type=PhaseType.ANNOTATE,
            model_type=GEMINI_PRO,  # Example: Use a different model
            # custom_output_path=Path("path/to/custom_annotated_output.md"),
        ),
        PhaseConfig(
            phase_type=PhaseType.FINAL,
            model_type=OPENAI_04_MINI,
        ),
        PhaseConfig(
            phase_type=PhaseType.FORMATTING,
            model_type=GEMINI_PRO,  # Example: Use a different model for formatting
        ),
    ]

    # Create a run configuration from the defined phases
    config = RunConfig(
        book_name="On Liberty",
        author_name="John Stuart Mill",
        input_file=Path(
            r"books\On Liberty\markdown\Mill, On Liberty\Mill, On Liberty Clean.md"
        ),
        output_dir=Path(r"books\On Liberty\markdown\Mill, On Liberty"),
        original_file=Path(
            r"books\On Liberty\markdown\Mill, On Liberty\Mill, On Liberty Clean.md"
        ),
        phases=run_phases,
    )

    # Run the pipeline
    run_pipeline(config)


if __name__ == "__main__":
    main()
