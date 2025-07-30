"""
Run command module.

This module handles running pipeline processing for books from markdown sources.
"""

import argparse
import importlib
import sys

from .common import get_books_with_run, validate_book_exists


def run_book(book_name: str) -> None:
    """
    Run a specific book by importing its run module and calling the main function.

    Args:
        book_name: Name of the book to run

    Raises:
        ImportError: If the book's run module cannot be imported
        AttributeError: If the book's run module doesn't have a main function
    """
    try:
        # Import the book's run module
        module_path = f"books.{book_name}.run"
        run_module = importlib.import_module(module_path)

        # Call the main function
        if hasattr(run_module, "main"):
            run_module.main()
        else:
            print(f"Error: Book '{book_name}' does not have a main function in its run.py module")
            sys.exit(1)

    except ImportError as e:
        print(f"Error: Could not import run module for book '{book_name}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running book '{book_name}': {e}")
        sys.exit(1)


def setup_run_parser(subparsers) -> argparse.ArgumentParser:
    """
    Set up the run command parser.

    Args:
        subparsers: The subparsers object to add the run command to

    Returns:
        The run command parser
    """
    available_books = get_books_with_run()
    
    parser = subparsers.add_parser(
        "run",
        help="Run pipeline processing for books from markdown sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available books for running:
{chr(10).join("  " + book for book in available_books) if available_books else "  No runnable books found"}

Examples:
  python -m cli run the_federalist_papers
  python -m cli run on_liberty
        """,
    )

    parser.add_argument("book_name", help="Name of the book to run (must match a directory in books/)")
    
    return parser


def handle_run_command(args) -> None:
    """
    Handle the run command.

    Args:
        args: Parsed command line arguments
    """
    available_books = get_books_with_run()
    matched_book_name = validate_book_exists(args.book_name, available_books)

    print(f"Running pipeline for book '{matched_book_name}'...")
    run_book(matched_book_name)