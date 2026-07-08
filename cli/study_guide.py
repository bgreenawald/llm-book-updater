from __future__ import annotations

from pathlib import Path

import click
from book_updater.study_guide import run_study_guide
from book_updater.study_guide_standalone import build_standalone_study_guide_config
from llm_core import DEEPSEEK_V4_PRO, ModelConfig, Provider


@click.command("study-guide")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, path_type=Path), help="Output directory.")
@click.option("--output-filename", default="study_guide.md", show_default=True, help="Final output filename.")
@click.option("--original-file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--book-name", help="Book/display name used in prompts. Defaults to the input filename.")
@click.option("--author-name", help='Author name used in prompts. Defaults to "Unknown Author".')
@click.option("--book-id", help="Identifier used in config metadata. Defaults to the input filename stem.")
@click.option(
    "--provider",
    default=DEEPSEEK_V4_PRO.provider.value,
    show_default=True,
    type=click.Choice([p.value for p in Provider]),
)
@click.option("--model", "model_id", default=DEEPSEEK_V4_PRO.model_id, show_default=True, help="Model ID.")
@click.option("--provider-model-name", help="Provider-specific model name, if different from --model.")
@click.option("--max-workers", type=int, help="Maximum concurrent workers for section processing.")
@click.option("--notes-system-prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--notes-user-prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--flashcards-system-prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--flashcards-user-prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.help_option("--help", "-h")
def study_guide_command(
    input_file: Path,
    output_dir: Path | None,
    output_filename: str,
    original_file: Path | None,
    book_name: str | None,
    author_name: str | None,
    book_id: str | None,
    provider: str,
    model_id: str,
    provider_model_name: str | None,
    max_workers: int | None,
    notes_system_prompt: Path | None,
    notes_user_prompt: Path | None,
    flashcards_system_prompt: Path | None,
    flashcards_user_prompt: Path | None,
) -> None:
    """Generate a study guide directly from a markdown input file."""
    model = ModelConfig(provider=Provider(provider), model_id=model_id, provider_model_name=provider_model_name)
    config = build_standalone_study_guide_config(
        input_file=input_file,
        output_dir=output_dir,
        original_file=original_file,
        book_name=book_name,
        author_name=author_name,
        book_id=book_id,
        model=model,
        max_workers=max_workers,
        output_filename=output_filename,
        notes_system_prompt=notes_system_prompt,
        notes_user_prompt=notes_user_prompt,
        flashcards_system_prompt=flashcards_system_prompt,
        flashcards_user_prompt=flashcards_user_prompt,
    )
    output_file = run_study_guide(config)
    click.echo(f"Study guide generated: {output_file}")
