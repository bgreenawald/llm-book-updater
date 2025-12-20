import sys
from pathlib import Path
from typing import List

from src.common.provider import Provider
from src.config import PhaseConfig, PhaseType, RunConfig
from src.llm_model import ModelConfig
from src.logging_config import setup_logging
from src.pipeline import run_pipeline

CHATGPT_GPT52 = ModelConfig(Provider.OPENAI, "gpt-5.2")
GROK_41 = ModelConfig(Provider.OPENROUTER, "x-ai/grok-4.1-fast")
GEMINI_PRO = ModelConfig(Provider.GEMINI, "gemini-3-pro-preview")
DEEPSEEK_V32 = ModelConfig(Provider.OPENROUTER, "deepseek/deepseek-v3.2")

run_phases: List[PhaseConfig] = [
    PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        model=GEMINI_PRO,
        reasoning={"effort": "high"},
        use_batch=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.EDIT,
        model=CHATGPT_GPT52,
        reasoning={"effort": "high"},
        use_batch=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.FINAL,
        model=CHATGPT_GPT52,
        reasoning={"effort": "high"},
        use_batch=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.INTRODUCTION,
        model=GROK_41,
        reasoning={"effort": "high"},
    ),
    PhaseConfig(
        phase_type=PhaseType.SUMMARY,
        model=DEEPSEEK_V32,
        reasoning={"effort": "high"},
        enable_retry=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.ANNOTATE,
        model=GROK_41,
        reasoning={"effort": "high"},
    ),
]

# Main configuration object for the pipeline run.
config = RunConfig(
    book_id="the_wealth_of_nations",
    book_name="The Wealth of Nations",
    author_name="Adam Smith",
    input_file=Path(r"books/the_wealth_of_nations/input_transformed.md"),
    output_dir=Path(r"books/the_wealth_of_nations/output"),
    original_file=Path(r"books/the_wealth_of_nations/input_transformed.md"),
    phases=run_phases,
    max_workers=10,
)


def main() -> None:
    """Main function to run the pipeline."""
    logger = setup_logging("the_wealth_of_nations")
    try:
        logger.info("Starting pipeline execution from main.py")
        run_pipeline(config=config)
        logger.success("Pipeline execution finished.")
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
