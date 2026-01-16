from __future__ import annotations

from pathlib import Path

from llm_core import LlmModel, ModelConfig
from llm_core.config import DEFAULT_TAGS_TO_PRESERVE, BaseConfig
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

    notes_factory_config = PhaseConfig(
        phase_type=config.notes_phase.phase_type,
        name="study_guide_notes",
        input_file_path=config.input_file,
        output_file_path=notes_output,
        original_file_path=config.original_file,
        system_prompt_path=config.notes_phase.system_prompt_path,
        user_prompt_path=config.notes_phase.user_prompt_path,
        book_name=config.book_name,
        author_name=config.author_name,
        llm_model_instance=notes_model,
        reasoning=config.notes_phase.reasoning,
        llm_kwargs=config.notes_phase.llm_kwargs,
        post_processors=config.notes_phase.post_processors,
        use_batch=config.notes_phase.use_batch,
        batch_size=config.notes_phase.batch_size,
        enable_retry=config.notes_phase.enable_retry,
        max_retries=config.notes_phase.max_retries,
        use_subblocks=config.notes_phase.use_subblocks,
        max_subblock_tokens=config.notes_phase.max_subblock_tokens,
        min_subblock_tokens=config.notes_phase.min_subblock_tokens,
        skip_if_less_than_tokens=config.notes_phase.skip_if_less_than_tokens,
    )

    flashcards_factory_config = PhaseConfig(
        phase_type=config.flashcards_phase.phase_type,
        name="study_guide_flashcards",
        input_file_path=config.input_file,
        output_file_path=flashcards_output,
        original_file_path=config.original_file,
        system_prompt_path=config.flashcards_phase.system_prompt_path,
        user_prompt_path=config.flashcards_phase.user_prompt_path,
        book_name=config.book_name,
        author_name=config.author_name,
        llm_model_instance=flashcards_model,
        reasoning=config.flashcards_phase.reasoning,
        llm_kwargs=config.flashcards_phase.llm_kwargs,
        post_processors=config.flashcards_phase.post_processors,
        use_batch=config.flashcards_phase.use_batch,
        batch_size=config.flashcards_phase.batch_size,
        enable_retry=config.flashcards_phase.enable_retry,
        max_retries=config.flashcards_phase.max_retries,
        use_subblocks=config.flashcards_phase.use_subblocks,
        max_subblock_tokens=config.flashcards_phase.max_subblock_tokens,
        min_subblock_tokens=config.flashcards_phase.min_subblock_tokens,
        skip_if_less_than_tokens=config.flashcards_phase.skip_if_less_than_tokens,
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
        hash_count = 0
        for char in section_header:
            if char == "#":
                hash_count += 1
            else:
                break
        if hash_count:
            hash_count = max(1, min(hash_count + 1, 6))
            return f"{'#' * hash_count} Flashcards"
    return "## Flashcards"
