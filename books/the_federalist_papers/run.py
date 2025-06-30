import sys
from pathlib import Path
from typing import List

from src.config import PhaseConfig, PhaseType, RunConfig
from src.llm_model import DEEPSEEK, GEMINI_FLASH, GEMINI_PRO
from src.logging_config import setup_logging
from src.pipeline import run_pipeline

run_phases: List[PhaseConfig] = [
    PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        model_type=GEMINI_FLASH,
        reasoning={"effort": "high"},
    ),
    PhaseConfig(
        phase_type=PhaseType.EDIT,
        model_type=GEMINI_PRO,
        reasoning={"effort": "high"},
    ),
    PhaseConfig(
        phase_type=PhaseType.FINAL,
        model_type=DEEPSEEK,
        reasoning={"effort": "high"},
    ),
    PhaseConfig(
        phase_type=PhaseType.INTRODUCTION,
        model_type=GEMINI_FLASH,
        reasoning={"effort": "high"},
    ),
    PhaseConfig(
        phase_type=PhaseType.SUMMARY,
        model_type=GEMINI_FLASH,
        reasoning={"effort": "high"},
    ),
    PhaseConfig(
        phase_type=PhaseType.ANNOTATE,
        model_type=GEMINI_FLASH,
        reasoning={"effort": "high"},
    ),
]

# Main configuration object for the pipeline run.
config = RunConfig(
    book_name="the_federalist_papers",
    author_name="Alexander Hamilton, James Madison, John Jay",
    input_file=Path(r"books/the_federalist_papers/input_small.md"),
    output_dir=Path(r"books/the_federalist_papers/output"),
    original_file=Path(r"books/the_federalist_papers/input_small.md"),
    phases=run_phases,
    length_reduction=(35, 50),
    max_workers=10,
)


def main() -> None:
    """Main function to run the pipeline."""
    logger = setup_logging("the_federalist_papers")
    try:
        logger.info("Starting pipeline execution from main.py")
        run_pipeline(config=config)
        logger.success("Pipeline execution finished.")
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
