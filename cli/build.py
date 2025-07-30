"""
Build command module.

This module handles building books from markdown sources to EPUB/PDF formats.
"""

import argparse
import importlib
import sys
from typing import List

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
            print(f"Error: Book '{book_name}' does not have a build function in its build.py module")
            sys.exit(1)

    except ImportError as e:
        print(f"Error: Could not import build module for book '{book_name}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error building book '{book_name}': {e}")
        sys.exit(1)


def setup_build_parser(subparsers) -> argparse.ArgumentParser:
    """
    Set up the build command parser.

    Args:
        subparsers: The subparsers object to add the build command to

    Returns:
        The build command parser
    """
    available_books = get_books_with_build()
    
    parser = subparsers.add_parser(
        "build",
        help="Build books from markdown sources to EPUB/PDF formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available books for building:
{chr(10).join("  " + book for book in available_books) if available_books else "  No buildable books found"}

Examples:
  python -m cli build the_federalist_papers v1.0.0
  python -m cli build on_liberty v0.1-alpha
        """,
    )

    parser.add_argument("book_name", help="Name of the book to build (must match a directory in books/)")
    parser.add_argument("version", help="Version string for the build (e.g., 'v1.0.0', 'v0.1-alpha')")
    
    return parser


def handle_build_command(args) -> None:
    """
    Handle the build command.

    Args:
        args: Parsed command line arguments
    """
    available_books = get_books_with_build()
    matched_book_name = validate_book_exists(args.book_name, available_books)

    print(f"Building book '{matched_book_name}' version '{args.version}'...")
    build_book(matched_book_name, args.version)