import sys
from pathlib import Path
from typing import List

from book_updater import PhaseConfig, PhaseType, RunConfig
from book_updater.logging_config import setup_logging
from book_updater.pipeline import run_pipeline
from llm_core import ModelConfig, Provider

GEMINI_3_PRO = ModelConfig(provider=Provider.GEMINI, model_id="gemini-3.1-pro-preview")
KIMI_K2 = ModelConfig(provider=Provider.OPENROUTER, model_id="moonshotai/kimi-k2-thinking")

# Input for the abridged pipeline: the FINAL_TWO_STAGE output from the main pipeline
_MODERNIZED_OUTPUT = Path("books/a_theory_of_justice/output/03-input_transformed Final_two_stage_1.md")

abridge_phases: List[PhaseConfig] = [
    PhaseConfig(
        phase_type=PhaseType.ABRIDGE_PLAN,
        model=GEMINI_3_PRO,
        reasoning={"effort": "high"},
        enable_retry=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.ABRIDGE_WRITE,
        model=KIMI_K2,
        reasoning={"effort": "high"},
        enable_retry=True,
    ),
]

config = RunConfig(
    book_id="a_theory_of_justice",
    book_name="A Theory of Justice",
    author_name="John Rawls",
    input_file=_MODERNIZED_OUTPUT,
    output_dir=Path(r"books/a_theory_of_justice/output"),
    original_file=_MODERNIZED_OUTPUT,
    phases=abridge_phases,
    max_workers=10,
)


def main() -> None:
    """Main function to run the abridged pipeline."""
    logger = setup_logging("a_theory_of_justice")
    try:
        logger.info("Starting abridged pipeline execution")
        run_pipeline(config=config)
        logger.success("Abridged pipeline execution finished.")
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
