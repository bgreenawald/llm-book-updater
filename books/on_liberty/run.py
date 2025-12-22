import sys
from pathlib import Path
from typing import List

from src.api.config import PhaseConfig, PhaseType, RunConfig, TwoStageModelConfig
from src.api.provider import Provider
from src.core.pipeline import run_pipeline
from src.models.model import ModelConfig
from src.utils.logging_config import setup_logging

DEEPSEEK_V32 = ModelConfig(provider=Provider.OPENROUTER, model_id="deepseek/deepseek-v3.2")
GEMINI_3_FLASH = ModelConfig(provider=Provider.GEMINI, model_id="gemini-3-flash-preview")
KIMI_K2 = ModelConfig(provider=Provider.OPENROUTER, model_id="moonshotai/kimi-k2-thinking")

run_phases: List[PhaseConfig] = [
    PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        model=GEMINI_3_FLASH,
        reasoning={"effort": "high"},
        enable_retry=True,
        min_subblock_tokens=4096,
        max_subblock_tokens=8192,
        use_subblocks=True,
        use_batch=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.EDIT,
        model=KIMI_K2,
        reasoning={"effort": "high"},
        enable_retry=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.FINAL_TWO_STAGE,
        two_stage_config=TwoStageModelConfig(
            identify_model=GEMINI_3_FLASH,
            implement_model=GEMINI_3_FLASH,
            identify_reasoning={"effort": "high"},
            implement_reasoning={"effort": "high"},
        ),
        enable_retry=True,
        use_batch=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.INTRODUCTION,
        model=DEEPSEEK_V32,
        reasoning={"effort": "high"},
        enable_retry=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.SUMMARY,
        model=DEEPSEEK_V32,
        reasoning={"effort": "high"},
        enable_retry=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.ANNOTATE,
        model=GEMINI_3_FLASH,
        reasoning={"effort": "high"},
        enable_retry=True,
        min_subblock_tokens=4096,
        max_subblock_tokens=8192,
        use_subblocks=True,
        use_batch=True,
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
