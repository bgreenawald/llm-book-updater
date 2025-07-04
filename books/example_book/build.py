"""
Example build script for "Example Book".

This demonstrates how to create a book-specific build script using the shared base builder.
"""

from pathlib import Path
from typing import Dict, Optional

from books.base_builder import BaseBookBuilder, BookConfig, auto_detect_book_name, create_build_parser


class ExampleBookBuilder(BaseBookBuilder):
    """
    Book-specific builder for Example Book.

    This class shows how to implement the required abstract methods
    for a book with different file naming conventions.
    """

    def get_source_files(self) -> Dict[str, Path]:
        """
        Get the source file paths for Example Book.

        This example shows how to handle different file naming patterns.
        """
        return {
            "modernized": self.config.source_output_dir / "final_modernized.md",
            "annotated": self.config.source_output_dir / "final_annotated.md",
        }

    def get_original_file(self) -> Optional[Path]:
        """
        Get the original source file for Example Book.

        This example shows how to handle a different original file naming pattern.
        """
        # Look for files starting with "original_" in the output directory
        if not self.config.source_output_dir.exists():
            return None

        for file_path in self.config.source_output_dir.glob("original_*"):
            if file_path.is_file():
                return file_path

        return None


def build(version: str, name: str):
    """
    Build function for Example Book.

    Args:
        version: Version string for the build
        name: Name of the book directory
    """
    # Create book configuration
    config = BookConfig(
        name=name,
        version=version,
        title="Example Book: A Demonstration",
        author="Example Author",
        # Optional: specify a custom clean title for filenames
        clean_title="Example-Book-Demonstration",
    )

    # Create and run the builder
    builder = ExampleBookBuilder(config=config)
    builder.build()


def main():
    """
    Main function to parse command-line arguments and run the build.
    """
    parser = create_build_parser()
    args = parser.parse_args()

    if args.command == "build":
        # Auto-detect book name if not provided
        book_name = args.name or auto_detect_book_name()
        build(args.version, book_name)


if __name__ == "__main__":
    main()
