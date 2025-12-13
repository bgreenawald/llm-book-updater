import sys
from pathlib import Path
from typing import List

from src.common.provider import Provider
from src.config import PhaseConfig, PhaseType, RunConfig
from src.llm_model import ModelConfig
from src.logging_config import setup_logging
from src.pipeline import run_pipeline

# Use OpenRouter for all phases
GOOGLE_GEMINI_PRO = ModelConfig(provider=Provider.GEMINI, model_id="gemini-3-pro-preview")
CHATGPT_GPT5_MINI = ModelConfig(provider=Provider.OPENAI, model_id="gpt-5-mini")
CHATGPT_GPT5 = ModelConfig(provider=Provider.OPENAI, model_id="gpt-5")
GROK_41 = ModelConfig(provider=Provider.OPENROUTER, model_id="x-ai/grok-4.1-fast")

run_phases: List[PhaseConfig] = [
    PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        model=GROK_41,
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
        model=GROK_41,
        reasoning={"effort": "medium"},
    ),
    PhaseConfig(
        phase_type=PhaseType.SUMMARY,
        model=GROK_41,
        reasoning={"effort": "medium"},
    ),
    PhaseConfig(
        phase_type=PhaseType.ANNOTATE,
        model=GROK_41,
        reasoning={"effort": "medium"},
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
