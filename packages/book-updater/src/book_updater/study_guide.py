from __future__ import annotations

from pathlib import Path

from llm_core import LlmModel, ModelConfig
from llm_core.config import DEFAULT_TAGS_TO_PRESERVE, BaseConfig
from loguru import logger
from pydantic import Field

from book_updater.config import PhaseConfig
from book_updater.phases.factory import PhaseFactory
from book_updater.phases.utils import extract_markdown_blocks, get_header_and_body


class StudyGuideConfig(BaseConfig):
    book_id: str
    book_name: str
    author_name: str
    input_file: Path
    output_dir: Path
    original_file: Path
    notes_phase: PhaseConfig
    flashcards_phase: PhaseConfig
    output_filename: str = "study_guide.md"
    notes_draft_filename: str = "study_guide_notes_draft.md"
    flashcards_draft_filename: str = "study_guide_flashcards_draft.md"
    max_workers: int | None = None
    tags_to_preserve: list[str] = Field(default_factory=lambda: list(DEFAULT_TAGS_TO_PRESERVE))


def run_study_guide(config: StudyGuideConfig) -> Path:
    """Run notes + flashcards phases and assemble final study guide."""
    if not config.input_file.exists():
        raise FileNotFoundError(f"Study guide input file not found: {config.input_file}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    notes_output = config.output_dir / config.notes_draft_filename
    flashcards_output = config.output_dir / config.flashcards_draft_filename

    model_cache: dict[tuple[str, str, str | None], LlmModel] = {}

    def _model_key(model: ModelConfig) -> tuple[str, str, str | None]:
        return (model.provider.value, model.model_id, model.provider_model_name)

    def _get_model(model_config: ModelConfig) -> LlmModel:
        key = _model_key(model_config)
        if key not in model_cache:
            model_cache[key] = LlmModel.create(model=model_config)
        return model_cache[key]

    notes_model = _get_model(config.notes_phase.model)
    flashcards_model = _get_model(config.flashcards_phase.model)

    notes_factory_config = config.notes_phase.model_copy(
        update={
            "name": "study_guide_notes",
            "input_file_path": config.input_file,
            "output_file_path": notes_output,
            "original_file_path": config.original_file,
            "book_name": config.book_name,
            "author_name": config.author_name,
            "llm_model_instance": notes_model,
        }
    )

    flashcards_factory_config = config.flashcards_phase.model_copy(
        update={
            "name": "study_guide_flashcards",
            "input_file_path": config.input_file,
            "output_file_path": flashcards_output,
            "original_file_path": config.original_file,
            "book_name": config.book_name,
            "author_name": config.author_name,
            "llm_model_instance": flashcards_model,
        }
    )

    try:
        notes_phase = PhaseFactory.create_standard_phase(
            config=notes_factory_config,
            tags_to_preserve=config.tags_to_preserve,
            max_workers=config.max_workers,
        )
        flashcards_phase = PhaseFactory.create_standard_phase(
            config=flashcards_factory_config,
            tags_to_preserve=config.tags_to_preserve,
            max_workers=config.max_workers,
        )

        notes_phase.run()
        flashcards_phase.run()
    finally:
        for model in model_cache.values():
            model.close()

    if not notes_output.exists():
        raise FileNotFoundError(f"Notes draft not found after phase run: {notes_output}")
    if not flashcards_output.exists():
        raise FileNotFoundError(f"Flashcards draft not found after phase run: {flashcards_output}")

    final_output = config.output_dir / config.output_filename
    assemble_study_guide(notes_file=notes_output, flashcards_file=flashcards_output, output_file=final_output)
    return final_output


def assemble_study_guide(notes_file: Path, flashcards_file: Path, output_file: Path) -> None:
    """Combine notes and flashcards drafts into a single study guide."""
    notes_text = notes_file.read_text(encoding="utf-8")
    flashcards_text = flashcards_file.read_text(encoding="utf-8")

    notes_blocks = extract_markdown_blocks(notes_text)
    flashcards_blocks = extract_markdown_blocks(flashcards_text)

    if len(notes_blocks) != len(flashcards_blocks):
        raise ValueError(
            "Study guide block counts do not match: "
            f"{notes_file} has {len(notes_blocks)} blocks, "
            f"{flashcards_file} has {len(flashcards_blocks)} blocks"
        )

    assembled_sections: list[str] = []
    for notes_block, flashcards_block in zip(notes_blocks, flashcards_blocks):
        notes_header, notes_body = get_header_and_body(notes_block)
        flashcards_header, flashcards_body = get_header_and_body(flashcards_block)

        if notes_header and flashcards_header and notes_header.strip() != flashcards_header.strip():
            logger.warning(f'Mismatched headers found.\n  Notes: "{notes_header}"\n  Flashcards: "{flashcards_header}"')
        section_header = notes_header or flashcards_header
        flashcards_heading = _flashcards_heading(section_header)

        section_parts = [
            section_header,
            "",
            notes_body,
            "",
            flashcards_heading,
            "",
            flashcards_body,
            "",
        ]
        assembled_sections.append("\n".join(section_parts))

    output_file.write_text("\n".join(assembled_sections).strip() + "\n", encoding="utf-8")


def _flashcards_heading(section_header: str) -> str:
    if section_header.startswith("#"):
        hash_count = len(section_header) - len(section_header.lstrip("#"))
        if hash_count > 0:
            # Increment heading level, capping at 6
            new_level = min(hash_count + 1, 6)
            return f"{'#' * new_level} Flashcards"
    return "## Flashcards"
