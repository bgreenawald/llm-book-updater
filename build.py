"""
Build command for the LLM Book Updater project.

This module provides a simplified build interface that can be run from the project root.
It dynamically discovers available books and provides a unified build command.

Usage:
    python -m build <book_name> <version>
    python -m build --help

Examples:
    python -m build the_federalist_papers v1.0.0
    python -m build on_liberty v0.1-alpha
"""

import argparse
import importlib
import sys
from pathlib import Path
from typing import List


def get_available_books() -> List[str]:
    """
    Dynamically discover available books by scanning the books directory.

    Returns:
        List of available book names (directory names that contain a build.py file)
    """
    books_dir = Path("books")
    if not books_dir.exists():
        return []

    available_books = []
    for item in books_dir.iterdir():
        if item.is_dir() and not item.name.startswith(".") and item.name != "__pycache__":
            # Check if the directory has a build.py file
            build_file = item / "build.py"
            if build_file.exists():
                available_books.append(item.name)

    return sorted(available_books)


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


def main():
    """
    Main function to parse command-line arguments and execute the build.
    """
    # Get available books for help text
    available_books = get_available_books()

    parser = argparse.ArgumentParser(
        description="Build books from markdown sources to EPUB/PDF formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available books:
{chr(10).join("  " + book for book in available_books) if available_books else "  No books found"}

Examples:
  python -m build the_federalist_papers v1.0.0
  python -m build on_liberty v0.1-alpha
        """,
    )

    parser.add_argument("book_name", help="Name of the book to build (must match a directory in books/)")

    parser.add_argument("version", help="Version string for the build (e.g., 'v1.0.0', 'v0.1-alpha')")

    args = parser.parse_args()

    # Validate that the book exists
    if args.book_name not in available_books:
        print(f"Error: Book '{args.book_name}' not found.")
        print(f"Available books: {', '.join(available_books)}")
        sys.exit(1)

    # Build the book
    print(f"Building book '{args.book_name}' version '{args.version}'...")
    build_book(args.book_name, args.version)


if __name__ == "__main__":
    main()
