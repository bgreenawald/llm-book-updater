import sys
from pathlib import Path
from typing import List

from src.common.provider import Provider
from src.config import PhaseConfig, PhaseType, RunConfig
from src.llm_model import ModelConfig
from src.logging_config import setup_logging
from src.pipeline import run_pipeline

# Use OpenRouter for all phases
GOOGLE_GEMINI_PRO = ModelConfig(Provider.GEMINI, "gemini-2.5-pro")
CHATGPT_GPT5_MINI = ModelConfig(Provider.OPENAI, "gpt-5-mini")
CHATGPT_GPT5 = ModelConfig(Provider.OPENAI, "gpt-5")

run_phases: List[PhaseConfig] = [
    PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        model=CHATGPT_GPT5_MINI,
        use_batch=True,
        reasoning={"effort": "medium"},
    ),
    PhaseConfig(
        phase_type=PhaseType.EDIT,
        model=GOOGLE_GEMINI_PRO,
        use_batch=True,
        reasoning={"effort": "medium"},
    ),
    PhaseConfig(
        phase_type=PhaseType.FINAL,
        model=CHATGPT_GPT5,
        use_batch=True,
        reasoning={"effort": "medium"},
    ),
    PhaseConfig(
        phase_type=PhaseType.INTRODUCTION,
        model=CHATGPT_GPT5_MINI,
        use_batch=True,
        reasoning={"effort": "medium"},
    ),
    PhaseConfig(
        phase_type=PhaseType.SUMMARY,
        model=CHATGPT_GPT5_MINI,
        use_batch=True,
        reasoning={"effort": "medium"},
    ),
    PhaseConfig(
        phase_type=PhaseType.ANNOTATE,
        model=CHATGPT_GPT5_MINI,
        use_batch=True,
        reasoning={"effort": "medium"},
    ),
]

# Main configuration object for the pipeline run.
config = RunConfig(
    book_id="the_federalist_papers",
    book_name="The Federalist Papers",
    author_name="Alexander Hamilton, James Madison, John Jay",
    input_file=Path(r"books/the_federalist_papers/input_transformed.md"),
    output_dir=Path(r"books/the_federalist_papers/output"),
    original_file=Path(r"books/the_federalist_papers/input_transformed.md"),
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
