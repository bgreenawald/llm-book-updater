import sys
from pathlib import Path
from typing import List

from src.common.provider import Provider
from src.config import PhaseConfig, PhaseType, RunConfig
from src.llm_model import ModelConfig
from src.logging_config import setup_logging
from src.pipeline import run_pipeline

# Use OpenRouter for all phases
OPENROUTER_GEMINI_FLASH_LITE = ModelConfig(Provider.OPENROUTER, "google/gemini-2.5-flash-lite")
GOOGLE_GEMINI_FLASH_LITE = ModelConfig(Provider.GEMINI, "gemini-2.5-flash-lite")
CHATGPT_GPT5_NANO = ModelConfig(Provider.OPENAI, "gpt-5-nano")

run_phases: List[PhaseConfig] = [
    PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        model=GOOGLE_GEMINI_FLASH_LITE,
        reasoning={"effort": "high"},
    ),
    PhaseConfig(
        phase_type=PhaseType.EDIT,
        model=CHATGPT_GPT5_NANO,
        reasoning={"effort": "high"},
    ),
    PhaseConfig(
        phase_type=PhaseType.FINAL,
        model=OPENROUTER_GEMINI_FLASH_LITE,
        reasoning={"effort": "high"},
    ),
    PhaseConfig(
        phase_type=PhaseType.INTRODUCTION,
        model=OPENROUTER_GEMINI_FLASH_LITE,
        reasoning={"effort": "high"},
    ),
    PhaseConfig(
        phase_type=PhaseType.SUMMARY,
        model=OPENROUTER_GEMINI_FLASH_LITE,
        reasoning={"effort": "high"},
    ),
    PhaseConfig(
        phase_type=PhaseType.ANNOTATE,
        model=OPENROUTER_GEMINI_FLASH_LITE,
        reasoning={"effort": "high"},
    ),
]

# Main configuration object for the pipeline run.
config = RunConfig(
    book_id="on_liberty",
    book_name="On Liberty",
    author_name="John Stuart Mill",
    input_file=Path(r"books/on_liberty/input_small.md"),
    output_dir=Path(r"books/on_liberty/output"),
    original_file=Path(r"books/on_liberty/input_small.md"),
    phases=run_phases,
    length_reduction=(35, 50),
    max_workers=10,
)


def main() -> None:
    """Main function to run the pipeline."""
    logger = setup_logging("on_liberty")
    try:
        logger.info("Starting pipeline execution from main.py")
        run_pipeline(config=config)
        logger.success("Pipeline execution finished.")
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
