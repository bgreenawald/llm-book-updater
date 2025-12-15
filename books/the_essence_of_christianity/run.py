import sys
from pathlib import Path
from typing import List

from src.common.provider import Provider
from src.config import PhaseConfig, PhaseType, RunConfig
from src.llm_model import ModelConfig
from src.logging_config import setup_logging
from src.pipeline import run_pipeline

# Model configurations for pipeline phases
GOOGLE_GEMINI_PRO = ModelConfig(Provider.GEMINI, "gemini-3-pro-preview")
CHATGPT_GPT5 = ModelConfig(Provider.OPENAI, "gpt-5.2")
GROK_41 = ModelConfig(Provider.OPENROUTER, "x-ai/grok-4.1-fast")

run_phases: List[PhaseConfig] = [
    PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        model=GROK_41,
        reasoning={"effort": "medium"},
    ),
    PhaseConfig(
        phase_type=PhaseType.EDIT,
        model=CHATGPT_GPT5,
        use_batch=True,
        reasoning={"effort": "medium"},
    ),
    PhaseConfig(
        phase_type=PhaseType.FINAL,
        model=GOOGLE_GEMINI_PRO,
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
    book_id="the_essence_of_christianity",
    book_name="The Essence of Christianity",
    author_name="Ludwig Feuerbach",
    input_file=Path(r"books/the_essence_of_christianity/input_transformed.md"),
    output_dir=Path(r"books/the_essence_of_christianity/output"),
    original_file=Path(r"books/the_essence_of_christianity/input_transformed.md"),
    phases=run_phases,
    length_reduction=(35, 50),
    max_workers=10,
)


def main() -> None:
    """Run the pipeline for The Essence of Christianity."""
    logger = setup_logging("the_essence_of_christianity")
    try:
        logger.info("Starting pipeline execution from run.py")
        run_pipeline(config=config)
        logger.success("Pipeline execution finished.")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"An error occurred during pipeline execution: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
