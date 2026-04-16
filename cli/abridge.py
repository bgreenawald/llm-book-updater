"""
Abridge command module.

This module handles running the abridged pipeline for books.
"""

import importlib
import sys

import click

from .common import get_books_with_abridge, validate_book_exists


def abridge_book(book_name: str, start_from_phase: int = 0) -> None:
    """
    Run the abridged pipeline for a specific book.

    Args:
        book_name: Name of the book to abridge
        start_from_phase: Phase index to start execution from (0-based)

    Raises:
        ImportError: If the book's abridge module cannot be imported
        AttributeError: If the book's abridge module doesn't have a main function
    """
    try:
        module_path = f"books.{book_name}.abridge"
        abridge_module = importlib.import_module(module_path)

        if hasattr(abridge_module, "config"):
            abridge_module.config.start_from_phase = start_from_phase
            if start_from_phase > 0:
                click.echo(f"Starting from phase {start_from_phase}")

        if hasattr(abridge_module, "main"):
            abridge_module.main()
        else:
            click.echo(f"Error: Book '{book_name}' does not have a main function in its abridge.py module")
            sys.exit(1)

    except ImportError as e:
        click.echo(f"Error: Could not import abridge module for book '{book_name}': {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error abridging book '{book_name}': {e}")
        sys.exit(1)


@click.command("abridge")
@click.argument("book_name", required=True)
@click.option(
    "--start-from-phase",
    type=int,
    default=0,
    help="Phase index to start execution from (0-based). Useful for resuming after a failure.",
)
@click.help_option("--help", "-h")
def abridge_command(book_name: str, start_from_phase: int) -> None:
    """
    Run the abridged pipeline for a book.

    BOOK_NAME: Name of the book to abridge (must match a directory in books/)

    Examples:
      python -m cli abridge on_liberty
      python -m cli abridge a_theory_of_justice
      python -m cli abridge on_liberty --start-from-phase 2
    """
    available_books = get_books_with_abridge()

    if not available_books:
        click.echo("No abridgeable books found")
        sys.exit(1)

    matched_book_name = validate_book_exists(book_name, available_books)

    click.echo(f"Running abridged pipeline for book '{matched_book_name}'...")
    abridge_book(matched_book_name, start_from_phase=start_from_phase)
