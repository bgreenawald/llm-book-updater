"""
Main CLI entry point for LLM Book Updater.

This module provides a unified command-line interface with subcommands for
different operations like building and running books.

Usage:
    python -m cli <command> [args...]
    python -m cli --help

Available commands:
    build                Build books from markdown sources to EPUB/PDF formats
    build abridged       Build books including the generated abridged edition
    run                  Run pipeline processing for books from markdown sources
    abridge              Run the abridged pipeline for a book
    consolidate-metadata Consolidate multiple metadata files into one
    cover                Generate book covers using AI image generation
    abridge-cover        Generate abridged edition covers from existing covers
    mini-cover           Generate mini covers (thumbnails) from existing covers
    release              Create a GitHub release for a built book

Examples:
    python -m cli build the_federalist_papers v1.0.0
    python -m cli build abridged the_federalist_papers v1.0.0
    python -m cli run on_liberty
    python -m cli abridge on_liberty
    python -m cli consolidate-metadata books/on_liberty/output
    python -m cli cover on_liberty
    python -m cli abridge-cover on_liberty
    python -m cli mini-cover on_liberty
    python -m cli release build/the_federalist_papers/v1.0
"""

import sys
from pathlib import Path

import click


def _add_workspace_package_paths() -> None:
    """Make workspace packages importable when the CLI is run from a source checkout."""
    repo_root = Path(__file__).resolve().parents[1]
    package_src_dirs = (
        repo_root / "packages" / "llm-core" / "src",
        repo_root / "packages" / "book-updater" / "src",
        repo_root / "packages" / "book-writer" / "src",
    )

    for src_dir in reversed(package_src_dirs):
        src_path = str(src_dir)
        if src_dir.is_dir() and src_path not in sys.path:
            sys.path.insert(0, src_path)


_add_workspace_package_paths()


@click.group()
@click.help_option("--help", "-h")
def cli():
    """LLM Book Updater - A tool for processing and building books from markdown sources."""
    pass


def _register_commands() -> None:
    """Register CLI subcommands after workspace packages have been made importable."""
    from .abridge import abridge_command
    from .abridge_cover import abridge_cover_command
    from .build import build_command
    from .consolidate import consolidate_command
    from .cover import cover_command
    from .mini_cover import mini_cover_command
    from .release import release_command
    from .run import run_command
    from .study_guide import study_guide_command

    cli.add_command(build_command)
    cli.add_command(run_command)
    cli.add_command(abridge_command)
    cli.add_command(consolidate_command)
    cli.add_command(cover_command)
    cli.add_command(abridge_cover_command)
    cli.add_command(mini_cover_command)
    cli.add_command(release_command)
    cli.add_command(study_guide_command)


_register_commands()


def main() -> None:
    """
    Main function to handle CLI execution with error handling.
    """
    try:
        cli()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
