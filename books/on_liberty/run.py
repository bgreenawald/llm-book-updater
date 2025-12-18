import sys
from pathlib import Path
from typing import List

from src.common.provider import Provider
from src.config import PhaseConfig, PhaseType, RunConfig
from src.llm_model import ModelConfig
from src.logging_config import setup_logging
from src.pipeline import run_pipeline

DEEPSEEK_V32 = ModelConfig(Provider.OPENROUTER, "deepseek/deepseek-v3.2")
GEMINI_3_FLASH = ModelConfig(Provider.OPENROUTER, "google/gemini-3-flash-preview")
KIMI_K2 = ModelConfig(Provider.OPENROUTER, "moonshotai/kimi-k2-thinking")

run_phases: List[PhaseConfig] = [
    PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        model=KIMI_K2,
        reasoning={"effort": "high"},
        enable_retry=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.EDIT,
        model=GEMINI_3_FLASH,
        reasoning={"effort": "high"},
        enable_retry=True,
        enabled=False,
    ),
    PhaseConfig(
        phase_type=PhaseType.INTRODUCTION,
        model=DEEPSEEK_V32,
        reasoning={"effort": "high"},
        enable_retry=True,
        enabled=False,
    ),
    PhaseConfig(
        phase_type=PhaseType.SUMMARY,
        model=DEEPSEEK_V32,
        reasoning={"effort": "high"},
        enable_retry=True,
        enabled=False,
    ),
]

# Main configuration object for the pipeline run.
config = RunConfig(
    book_id="on_liberty",
    book_name="On Liberty",
    author_name="John Stuart Mill",
    input_file=Path(r"books/on_liberty/input_transformed.md"),
    output_dir=Path(r"books/on_liberty/output"),
    original_file=Path(r"books/on_liberty/input_transformed.md"),
    phases=run_phases,
    length_reduction=(20, 30),
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
