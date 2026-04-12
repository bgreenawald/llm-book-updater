import sys
from pathlib import Path
from typing import List

from book_updater import PhaseConfig, PhaseType, RunConfig, TwoStageModelConfig
from book_updater.logging_config import setup_logging
from book_updater.pipeline import run_pipeline
from llm_core import ModelConfig, Provider

DEEPSEEK_V32 = ModelConfig(provider=Provider.OPENROUTER, model_id="deepseek/deepseek-v3.2")
GEMINI_3_FLASH = ModelConfig(provider=Provider.GEMINI, model_id="gemini-3-flash-preview")
KIMI_K2 = ModelConfig(provider=Provider.OPENROUTER, model_id="moonshotai/kimi-k2.5")
SONNET46 = ModelConfig(provider=Provider.OPENROUTER, model_id="anthropic/claude-sonnet-4.6")

run_phases: List[PhaseConfig] = [
    PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        model=SONNET46,
        reasoning={"effort": "medium"},
        enable_retry=True,
        min_subblock_tokens=4096,
        max_subblock_tokens=8192,
        use_subblocks=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.EDIT,
        model=SONNET46,
        reasoning={"effort": "medium"},
    ),
    PhaseConfig(
        phase_type=PhaseType.FINAL_TWO_STAGE,
        two_stage_config=TwoStageModelConfig(
            identify_model=SONNET46,
            implement_model=SONNET46,
            identify_reasoning={"effort": "medium"},
            implement_reasoning={"effort": "medium"},
        ),
        enable_retry=True,
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
        model=DEEPSEEK_V32,
        reasoning={"effort": "high"},
        enable_retry=True,
        min_subblock_tokens=4096,
        max_subblock_tokens=8192,
        use_subblocks=True,
    ),
]

# Main configuration object for the pipeline run.
config = RunConfig(
    book_id="death_of_the_author",
    book_name="The Death of the Author",
    author_name="Roland Barthes",
    input_file=Path(r"books/death_of_the_author/input_transformed.md"),
    output_dir=Path(r"books/death_of_the_author/output"),
    original_file=Path(r"books/death_of_the_author/input_transformed.md"),
    phases=run_phases,
    max_workers=10,
)


def main() -> None:
    """Main function to run the pipeline."""
    logger = setup_logging("death_of_the_author")
    try:
        logger.info("Starting pipeline execution from main.py")
        run_pipeline(config=config)
        logger.success("Pipeline execution finished.")
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
