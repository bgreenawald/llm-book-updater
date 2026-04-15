import sys
from pathlib import Path
from typing import List

from book_updater import PhaseConfig, PhaseType, RunConfig
from book_updater.logging_config import setup_logging
from book_updater.pipeline import run_pipeline
from llm_core import ModelConfig, Provider

GEMINI_3_FLASH = ModelConfig(provider=Provider.GEMINI, model_id="gemini-3-flash-preview")

# Input for the abridged pipeline: the FINAL_TWO_STAGE output from the main pipeline
_MODERNIZED_OUTPUT = Path("books/the_communist_manifesto/output/03-input_transformed Final_two_stage_1.md")

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
    book_id="the_communist_manifesto",
    book_name="The Communist Manifesto",
    author_name="Karl Marx and Friedrich Engels",
    input_file=_MODERNIZED_OUTPUT,
    output_dir=Path(r"books/the_communist_manifesto/output"),
    original_file=_MODERNIZED_OUTPUT,
    phases=abridge_phases,
    max_workers=10,
)


def main() -> None:
    """Main function to run the abridged pipeline."""
    logger = setup_logging("the_communist_manifesto")
    try:
        logger.info("Starting abridged pipeline execution")
        run_pipeline(config=config)
        logger.success("Abridged pipeline execution finished.")
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
