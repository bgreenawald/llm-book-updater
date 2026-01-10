"""
Main CLI entry point for LLM Book Updater.

This module provides a unified command-line interface with subcommands for
different operations like building and running books.

Usage:
    python -m cli <command> [args...]
    python -m cli --help

Available commands:
    build                Build books from markdown sources to EPUB/PDF formats
    run                  Run pipeline processing for books from markdown sources
    consolidate-metadata Consolidate multiple metadata files into one
    release              Create a GitHub release for a built book

Examples:
    python -m cli build the_federalist_papers v1.0.0
    python -m cli run on_liberty
    python -m cli consolidate-metadata books/on_liberty/output
    python -m cli release build/the_federalist_papers/v1.0
"""

import sys

import click

from .build import build_command
from .consolidate import consolidate_command
from .release import release_command
from .run import run_command


@click.group()
@click.help_option("--help", "-h")
def cli():
    """LLM Book Updater - A tool for processing and building books from markdown sources."""
    pass


# Add the subcommands
cli.add_command(build_command)
cli.add_command(run_command)
cli.add_command(consolidate_command)
cli.add_command(release_command)


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
