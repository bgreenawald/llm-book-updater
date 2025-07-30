"""
Main CLI entry point for LLM Book Updater.

This module provides a unified command-line interface with subcommands for
different operations like building and running books.

Usage:
    python -m cli <command> [args...]
    python -m cli --help

Available commands:
    build   Build books from markdown sources to EPUB/PDF formats
    run     Run pipeline processing for books from markdown sources

Examples:
    python -m cli build the_federalist_papers v1.0.0
    python -m cli run on_liberty
"""

import sys

import click

from .build import build_command
from .run import run_command


@click.group()
@click.help_option("--help", "-h")
def cli():
    """LLM Book Updater - A tool for processing and building books from markdown sources."""
    pass


# Add the subcommands
cli.add_command(build_command)
cli.add_command(run_command)


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
