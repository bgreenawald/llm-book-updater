import sys
from pathlib import Path
from typing import List

from book_updater import PhaseConfig, PhaseType, RunConfig
from book_updater.logging_config import setup_logging
from book_updater.pipeline import run_pipeline
from llm_core import ModelConfig, Provider

GEMINI_3_FLASH = ModelConfig(provider=Provider.GEMINI, model_id="gemini-3-flash-preview")

# Input for the abridged pipeline: the FINAL_TWO_STAGE output from the main pipeline
_MODERNIZED_OUTPUT = Path("books/the_wealth_of_nations/output/03-input_transformed Final_two_stage_1.md")

abridge_phases: List[PhaseConfig] = [
    PhaseConfig(
        phase_type=PhaseType.ABRIDGE_PLAN,
        model=GEMINI_3_FLASH,
        reasoning={"effort": "high"},
        enable_retry=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.ABRIDGE_WRITE,
        model=GEMINI_3_FLASH,
        reasoning={"effort": "high"},
        enable_retry=True,
    ),
]

config = RunConfig(
    book_id="the_wealth_of_nations",
    book_name="The Wealth of Nations",
    author_name="Adam Smith",
    input_file=_MODERNIZED_OUTPUT,
    output_dir=Path(r"books/the_wealth_of_nations/output"),
    original_file=_MODERNIZED_OUTPUT,
    phases=abridge_phases,
    max_workers=3,
)


def main() -> None:
    """Main function to run the abridged pipeline."""
    logger = setup_logging("the_wealth_of_nations")
    try:
        logger.info("Starting abridged pipeline execution")
        run_pipeline(config=config)
        logger.success("Abridged pipeline execution finished.")
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
