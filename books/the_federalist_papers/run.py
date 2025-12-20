import sys
from pathlib import Path
from typing import List

from src.common.provider import Provider
from src.config import PhaseConfig, PhaseType, RunConfig, TwoStageModelConfig
from src.constants import LLM_DEFAULT_TEMPERATURE
from src.llm_model import ModelConfig
from src.logging_config import setup_logging
from src.pipeline import run_pipeline

DEEPSEEK_V32 = ModelConfig(Provider.OPENROUTER, "deepseek/deepseek-v3.2")
GEMINI_3_FLASH = ModelConfig(Provider.GEMINI, "gemini-3-flash-preview")
GPT_52 = ModelConfig(Provider.OPENAI, "gpt-5.2")

run_phases: List[PhaseConfig] = [
    PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        model=GEMINI_3_FLASH,
        reasoning={"effort": "high"},
        enable_retry=True,
        min_subblock_tokens=2048,
        max_subblock_tokens=4096,
        use_subblocks=True,
        temperature=LLM_DEFAULT_TEMPERATURE,
        use_batch=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.EDIT,
        model=GPT_52,
        reasoning={"effort": "high"},
        enable_retry=True,
        temperature=LLM_DEFAULT_TEMPERATURE,
        use_batch=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.FINAL_TWO_STAGE,
        model=GEMINI_3_FLASH,
        two_stage_config=TwoStageModelConfig(
            identify_model=GEMINI_3_FLASH,
            implement_model=GEMINI_3_FLASH,
            identify_temperature=LLM_DEFAULT_TEMPERATURE,
            implement_temperature=LLM_DEFAULT_TEMPERATURE,
        ),
        reasoning={"effort": "high"},
        enable_retry=True,
        temperature=LLM_DEFAULT_TEMPERATURE,
        use_batch=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.INTRODUCTION,
        model=DEEPSEEK_V32,
        reasoning={"effort": "high"},
        enable_retry=True,
        temperature=LLM_DEFAULT_TEMPERATURE,
    ),
    PhaseConfig(
        phase_type=PhaseType.SUMMARY,
        model=DEEPSEEK_V32,
        reasoning={"effort": "high"},
        enable_retry=True,
        temperature=LLM_DEFAULT_TEMPERATURE,
    ),
    PhaseConfig(
        phase_type=PhaseType.ANNOTATE,
        model=GEMINI_3_FLASH,
        reasoning={"effort": "high"},
        enable_retry=True,
        min_subblock_tokens=4096,
        max_subblock_tokens=8192,
        use_subblocks=True,
        temperature=LLM_DEFAULT_TEMPERATURE,
        use_batch=True,
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
    length_reduction=(50, 75),
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
