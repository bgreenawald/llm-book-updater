"""
Build command module.

This module handles building books from markdown sources to EPUB/PDF formats.
"""

import importlib
import sys

import click

from .common import get_books_with_build, validate_book_exists


def build_book(book_name: str, version: str) -> None:
    """
    Build a specific book by importing its build module and calling the build function.

    Args:
        book_name: Name of the book to build
        version: Version string for the build

    Raises:
        ImportError: If the book's build module cannot be imported
        AttributeError: If the book's build module doesn't have a build function
    """
    try:
        # Import the book's build module
        module_path = f"books.{book_name}.build"
        build_module = importlib.import_module(module_path)

        # Call the build function
        if hasattr(build_module, "build"):
            build_module.build(version, book_name)
        else:
            click.echo(f"Error: Book '{book_name}' does not have a build function in its build.py module")
            sys.exit(1)

    except ImportError as e:
        click.echo(f"Error: Could not import build module for book '{book_name}': {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error building book '{book_name}': {e}")
        sys.exit(1)


@click.command("build")
@click.argument("book_name", required=True)
@click.argument("version", required=True)
@click.help_option("--help", "-h")
def build_command(book_name: str, version: str) -> None:
    """
    Build books from markdown sources to EPUB/PDF formats.

    BOOK_NAME: Name of the book to build (must match a directory in books/)
    VERSION: Version string for the build (e.g., 'v1.0.0', 'v0.1-alpha')

    Examples:
      python -m cli build the_federalist_papers v1.0.0
      python -m cli build on_liberty v0.1-alpha
    """
    available_books = get_books_with_build()

    if not available_books:
        click.echo("No buildable books found")
        sys.exit(1)

    matched_book_name = validate_book_exists(book_name, available_books)

    click.echo(f"Building book '{matched_book_name}' version '{version}'...")
    build_book(matched_book_name, version)
