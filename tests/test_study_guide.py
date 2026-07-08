from pathlib import Path

from book_updater.config import PhaseConfig, PhaseType, PostProcessorType
from book_updater.study_guide import (
    STUDY_GUIDE_POST_PROCESSORS,
    _prepare_study_guide_phase_config,
    assemble_study_guide,
)
from llm_core import GEMINI_FLASH


def _phase_config(post_processors=None) -> PhaseConfig:
    return PhaseConfig(
        phase_type=PhaseType.MODERNIZE,
        model=GEMINI_FLASH,
        system_prompt_path=Path("prompts/notes_system.md"),
        user_prompt_path=Path("prompts/notes_user.md"),
        post_processors=post_processors,
    )


def test_study_guide_default_processors_allow_generated_headers():
    config = _phase_config()

    prepared_config = _prepare_study_guide_phase_config(config)

    assert prepared_config.post_processors == STUDY_GUIDE_POST_PROCESSORS
    assert PostProcessorType.NO_NEW_HEADERS not in prepared_config.post_processors


def test_study_guide_filters_no_new_headers_from_custom_processors():
    config = _phase_config(
        post_processors=[
            PostProcessorType.VALIDATE_NON_EMPTY_SECTION,
            PostProcessorType.NO_NEW_HEADERS,
            "no_new_headers",
            "remove_xml_tags",
        ]
    )

    prepared_config = _prepare_study_guide_phase_config(config)

    assert prepared_config.post_processors == [
        PostProcessorType.VALIDATE_NON_EMPTY_SECTION,
        "remove_xml_tags",
    ]


def test_assemble_study_guide_preserves_notes_subheadings(tmp_path):
    notes_file = tmp_path / "notes.md"
    flashcards_file = tmp_path / "flashcards.md"
    output_file = tmp_path / "study_guide.md"
    notes_file.write_text("# Section\n\n## Key Ideas\n\nImportant notes.\n", encoding="utf-8")
    flashcards_file.write_text("# Section\n\n## Review Cards\n\n- Q: What matters?\n  A: Headers.\n", encoding="utf-8")

    assemble_study_guide(notes_file=notes_file, flashcards_file=flashcards_file, output_file=output_file)

    output = output_file.read_text(encoding="utf-8")
    assert "## Key Ideas" in output
    assert "## Flashcards" in output
    assert "## Review Cards" in output
