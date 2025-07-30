"""
Shared utilities for CLI commands.

This module provides common functionality used across different CLI commands,
such as book discovery and validation.
"""

import sys
from pathlib import Path
from typing import List


def get_available_books() -> List[str]:
    """
    Dynamically discover available books by scanning the books directory.

    Returns:
        List of available book names (directory names that are valid book projects)
    """
    books_dir = Path("books")
    if not books_dir.exists():
        return []

    available_books = []
    for item in books_dir.iterdir():
        if item.is_dir() and not item.name.startswith(".") and item.name != "__pycache__":
            available_books.append(item.name)

    return sorted(available_books)


def get_books_with_build() -> List[str]:
    """
    Get books that have a build.py file.

    Returns:
        List of book names that can be built
    """
    books_dir = Path("books")
    if not books_dir.exists():
        return []

    available_books = []
    for item in books_dir.iterdir():
        if item.is_dir() and not item.name.startswith(".") and item.name != "__pycache__":
            build_file = item / "build.py"
            if build_file.exists():
                available_books.append(item.name)

    return sorted(available_books)


def get_books_with_run() -> List[str]:
    """
    Get books that have a run.py file.

    Returns:
        List of book names that can be run
    """
    books_dir = Path("books")
    if not books_dir.exists():
        return []

    available_books = []
    for item in books_dir.iterdir():
        if item.is_dir() and not item.name.startswith(".") and item.name != "__pycache__":
            run_file = item / "run.py"
            if run_file.exists():
                available_books.append(item.name)

    return sorted(available_books)


def find_matching_book(partial_name: str, available_books: List[str]) -> str:
    """
    Find a book that matches the partial name. If there's exactly one match, return it.
    If there are multiple matches or no matches, raise an error.

    Args:
        partial_name: Partial book name to match
        available_books: List of available book names

    Returns:
        The full book name that matches the partial name

    Raises:
        SystemExit: If no unique match is found
    """
    # First, check for exact match
    if partial_name in available_books:
        return partial_name
    
    # Find all books that start with the partial name (highest priority)
    prefix_matches = [book for book in available_books if book.startswith(partial_name)]
    
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    elif len(prefix_matches) > 1:
        print(f"Error: Multiple books match '{partial_name}':")
        for match in prefix_matches:
            print(f"  {match}")
        print("Please be more specific.")
        sys.exit(1)
    
    # If no prefix matches, find all books that contain the partial name as a substring
    substring_matches = [book for book in available_books if partial_name in book]
    
    if len(substring_matches) == 0:
        print(f"Error: No book found matching '{partial_name}'.")
        print(f"Available books: {', '.join(available_books)}")
        sys.exit(1)
    elif len(substring_matches) == 1:
        return substring_matches[0]
    else:
        print(f"Error: Multiple books match '{partial_name}':")
        for match in substring_matches:
            print(f"  {match}")
        print("Please be more specific.")
        sys.exit(1)


def validate_book_exists(book_name: str, available_books: List[str]) -> str:
    """
    Validate that a book exists in the available books list, supporting partial name matching.

    Args:
        book_name: Name of the book to validate (can be partial)
        available_books: List of available book names

    Returns:
        The full book name that matches

    Raises:
        SystemExit: If the book is not found or multiple matches exist
    """
    return find_matching_book(book_name, available_books)