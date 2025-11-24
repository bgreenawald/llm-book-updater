"""
Run command module.

This module handles running pipeline processing for books from markdown sources.
"""

import importlib
import sys

import click

from .common import get_books_with_run, validate_book_exists


def run_book(book_name: str, start_from_phase: int = 0) -> None:
    """
    Run a specific book by importing its run module and calling the main function.

    Args:
        book_name: Name of the book to run
        start_from_phase: Phase index to start execution from (0-based)

    Raises:
        ImportError: If the book's run module cannot be imported
        AttributeError: If the book's run module doesn't have a main function
    """
    try:
        # Import the book's run module
        module_path = f"books.{book_name}.run"
        run_module = importlib.import_module(module_path)

        # Modify the config in the run module to set start_from_phase
        if hasattr(run_module, "config"):
            run_module.config.start_from_phase = start_from_phase
            if start_from_phase > 0:
                click.echo(f"Starting from phase {start_from_phase}")

        # Call the main function
        if hasattr(run_module, "main"):
            run_module.main()
        else:
            click.echo(f"Error: Book '{book_name}' does not have a main function in its run.py module")
            sys.exit(1)

    except ImportError as e:
        click.echo(f"Error: Could not import run module for book '{book_name}': {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error running book '{book_name}': {e}")
        sys.exit(1)


@click.command("run")
@click.argument("book_name", required=True)
@click.option(
    "--start-from-phase",
    type=int,
    default=0,
    help="Phase index to start execution from (0-based). Useful for resuming after a failure.",
)
@click.help_option("--help", "-h")
def run_command(book_name: str, start_from_phase: int) -> None:
    """
    Run pipeline processing for books from markdown sources.

    BOOK_NAME: Name of the book to run (must match a directory in books/)

    Examples:
      python -m cli run the_federalist_papers
      python -m cli run on_liberty
      python -m cli run on_liberty --start-from-phase 3
    """
    available_books = get_books_with_run()

    if not available_books:
        click.echo("No runnable books found")
        sys.exit(1)

    matched_book_name = validate_book_exists(book_name, available_books)

    click.echo(f"Running pipeline for book '{matched_book_name}'...")
    run_book(matched_book_name, start_from_phase=start_from_phase)
