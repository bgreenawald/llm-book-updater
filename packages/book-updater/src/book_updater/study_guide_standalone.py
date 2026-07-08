"""Standalone study-guide generation from a single markdown input file.

Usage:
    uv run python -m cli study-guide path/to/input.md
    uv run python -m cli study-guide path/to/input.md --book-name "On Liberty" --author-name "John Stuart Mill"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_core import DEEPSEEK_V4_PRO, ModelConfig, Provider

from book_updater.config import PhaseConfig, PhaseType
from book_updater.study_guide import StudyGuideConfig, run_study_guide


def build_standalone_study_guide_config(
    *,
    input_file: Path,
    output_dir: Path | None = None,
    original_file: Path | None = None,
    book_name: str | None = None,
    author_name: str | None = None,
    book_id: str | None = None,
    model: ModelConfig = DEEPSEEK_V4_PRO,
    max_workers: int | None = None,
    output_filename: str = "study_guide.md",
    notes_system_prompt: Path | None = None,
    notes_user_prompt: Path | None = None,
    flashcards_system_prompt: Path | None = None,
    flashcards_user_prompt: Path | None = None,
) -> StudyGuideConfig:
    """Create a study-guide config from a single markdown input file."""
    input_file = input_file.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[4]
    prompts_dir = repo_root / "prompts"

    return StudyGuideConfig(
        book_id=book_id or input_file.stem,
        book_name=book_name or input_file.stem.replace("_", " ").replace("-", " ").title(),
        author_name=author_name or "Unknown Author",
        input_file=input_file,
        output_dir=(output_dir or input_file.parent / "study_guide").expanduser().resolve(),
        original_file=(original_file or input_file).expanduser().resolve(),
        output_filename=output_filename,
        notes_phase=PhaseConfig(
            phase_type=PhaseType.MODERNIZE,
            model=model,
            system_prompt_path=notes_system_prompt or prompts_dir / "notes_system.md",
            user_prompt_path=notes_user_prompt or prompts_dir / "notes_user.md",
            enable_retry=True,
        ),
        flashcards_phase=PhaseConfig(
            phase_type=PhaseType.MODERNIZE,
            model=model,
            system_prompt_path=flashcards_system_prompt or prompts_dir / "flashcards_system.md",
            user_prompt_path=flashcards_user_prompt or prompts_dir / "flashcards_user.md",
            enable_retry=True,
        ),
        max_workers=max_workers,
    )


def _parse_model(provider: str, model_id: str, provider_model_name: str | None) -> ModelConfig:
    try:
        provider_value = Provider(provider)
    except ValueError as exc:
        valid_providers = ", ".join(item.value for item in Provider)
        raise argparse.ArgumentTypeError(f"provider must be one of: {valid_providers}") from exc
    return ModelConfig(provider=provider_value, model_id=model_id, provider_model_name=provider_model_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a study guide from a standalone markdown input file.",
    )
    parser.add_argument("input_file", type=Path, help="Markdown file to turn into a study guide.")
    parser.add_argument("-o", "--output-dir", type=Path, help="Directory for drafts and the final study guide.")
    parser.add_argument("--output-filename", default="study_guide.md", help="Final output filename.")
    parser.add_argument("--original-file", type=Path, help="Original source file for context. Defaults to input_file.")
    parser.add_argument("--book-name", help="Book/display name used in prompts. Defaults to the input filename.")
    parser.add_argument("--author-name", help='Author name used in prompts. Defaults to "Unknown Author".')
    parser.add_argument("--book-id", help="Identifier used in config metadata. Defaults to the input filename stem.")
    parser.add_argument("--provider", default=DEEPSEEK_V4_PRO.provider.value, choices=[item.value for item in Provider])
    parser.add_argument("--model", default=DEEPSEEK_V4_PRO.model_id, help="Model ID to use for notes and flashcards.")
    parser.add_argument("--provider-model-name", help="Provider-specific model name, if different from --model.")
    parser.add_argument("--max-workers", type=int, help="Maximum concurrent workers for section processing.")
    parser.add_argument("--notes-system-prompt", type=Path, help="Override the notes system prompt path.")
    parser.add_argument("--notes-user-prompt", type=Path, help="Override the notes user prompt path.")
    parser.add_argument("--flashcards-system-prompt", type=Path, help="Override the flashcards system prompt path.")
    parser.add_argument("--flashcards-user-prompt", type=Path, help="Override the flashcards user prompt path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model = _parse_model(args.provider, args.model, args.provider_model_name)
    config = build_standalone_study_guide_config(
        input_file=args.input_file,
        output_dir=args.output_dir,
        original_file=args.original_file,
        book_name=args.book_name,
        author_name=args.author_name,
        book_id=args.book_id,
        model=model,
        max_workers=args.max_workers,
        output_filename=args.output_filename,
        notes_system_prompt=args.notes_system_prompt,
        notes_user_prompt=args.notes_user_prompt,
        flashcards_system_prompt=args.flashcards_system_prompt,
        flashcards_user_prompt=args.flashcards_user_prompt,
    )
    output_file = run_study_guide(config)
    print(f"Study guide generated: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
