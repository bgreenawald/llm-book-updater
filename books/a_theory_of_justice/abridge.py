import sys
from pathlib import Path
from typing import List

from book_updater import AbridgeWriteModelConfig, PhaseConfig, PhaseType, RunConfig
from book_updater.logging_config import setup_logging
from book_updater.pipeline import run_pipeline
from llm_core import ModelConfig, Provider

GPT_55 = ModelConfig(provider=Provider.OPENROUTER, model_id="openai/gpt-5.5")
KIMI_K2 = ModelConfig(provider=Provider.OPENROUTER, model_id="moonshotai/kimi-k2-thinking")
KIMI_K2_6 = ModelConfig(provider=Provider.OPENROUTER, model_id="moonshotai/kimi-k2.6")

# Input for the abridged pipeline: the Modernize output from the main pipeline
_MODERNIZED_OUTPUT = Path("books/a_theory_of_justice/output/01-input_transformed Modernize_1.md")

abridge_phases: List[PhaseConfig] = [
    PhaseConfig(
        phase_type=PhaseType.ABRIDGE_PLAN,
        model=GPT_55,
        reasoning={"effort": "medium"},
        enable_retry=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.ABRIDGE_FLESH,
        model=KIMI_K2_6,
        reasoning={"effort": "high"},
        enable_retry=True,
    ),
    PhaseConfig(
        phase_type=PhaseType.ABRIDGE_WRITE,
        abridge_write_config=AbridgeWriteModelConfig(
            profile_model=GPT_55,
            write_model=KIMI_K2,
            profile_reasoning={"effort": "low"},
            write_reasoning={"effort": "high"},
        ),
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
