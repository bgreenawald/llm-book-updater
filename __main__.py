"""
Entry point for running the build module directly.

This allows the build command to be run as:
    python -m build <book_name> <version>
"""

from build import main

if __name__ == "__main__":
    main()