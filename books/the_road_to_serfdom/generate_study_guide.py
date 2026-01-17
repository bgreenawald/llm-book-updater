"""Generate a study guide for The Use of Knowledge in Society.

This script creates a study guide with:
- Section-by-section notes
- Flashcards for key concepts

The output combines both into a single markdown file.
"""

import sys
from pathlib import Path

from book_updater import PhaseConfig, PhaseType
from book_updater.config import PostProcessorType
from book_updater.logging_config import setup_logging
from book_updater.study_guide import StudyGuideConfig, run_study_guide
from llm_core import ModelConfig, Provider

# Model configurations
GEMINI_FLASH = ModelConfig(provider=Provider.GEMINI, model_id="gemini-2.5-flash")
DEEPSEEK = ModelConfig(provider=Provider.OPENROUTER, model_id="deepseek/deepseek-v3.2")
# Study guide configuration
config = StudyGuideConfig(
    book_id="the_road_to_serfdom",
    book_name="The Road to Serfdom",
    author_name="Friedrich Hayek",
    input_file=Path("books/the_road_to_serfdom/output/03-input_transformed Final_two_stage_1.md"),
    output_dir=Path("books/the_road_to_serfdom/output/study_guide"),
    original_file=Path("books/the_road_to_serfdom/input_transformed.md"),
    notes_phase=PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        model=DEEPSEEK,
        system_prompt_path=Path("prompts/notes_system.md"),
        user_prompt_path=Path("prompts/notes_user.md"),
        enable_retry=True,
    ),
    flashcards_phase=PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        model=DEEPSEEK,
        system_prompt_path=Path("prompts/flashcards_system.md"),
        user_prompt_path=Path("prompts/flashcards_user.md"),
        post_processors=[
            PostProcessorType.VALIDATE_NON_EMPTY_SECTION,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE,
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.PRESERVE_F_STRING_TAGS,
        ],
        enable_retry=True,
    ),
    max_workers=5,
)


def main() -> None:
    """Main function to generate the study guide."""
    logger = setup_logging("study_guide_the_road_to_serfdom")
    try:
        logger.info("Starting study guide generation")
        output_file = run_study_guide(config=config)
        logger.success(f"Study guide generated successfully: {output_file}")
    except Exception as e:
        logger.error(f"An error occurred during study guide generation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
