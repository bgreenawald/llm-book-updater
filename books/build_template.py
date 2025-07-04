"""
Template for creating book-specific build scripts.

This template shows how to create a build script for a specific book
by extending the BaseBookBuilder class and implementing the required methods.
"""

from pathlib import Path
from typing import Dict, Optional

from books.base_builder import BaseBookBuilder, BookConfig, auto_detect_book_name, create_build_parser


class YourBookBuilder(BaseBookBuilder):
    """
    Book-specific builder for [Your Book Name].

    This class implements the abstract methods required by BaseBookBuilder
    to specify where the source files are located for this particular book.
    """

    def get_source_files(self) -> Dict[str, Path]:
        """
        Get the source file paths for this specific book.

        Override this method to specify where your book's source files are located.
        The file naming may vary between different books.

        Returns:
            Dictionary mapping file types to their source paths
        """
        return {
            "modernized": self.config.source_output_dir / "03-input_small Final_1.md",
            "annotated": self.config.source_output_dir / "06-input_small Annotate_1.md",
        }

    def get_original_file(self) -> Optional[Path]:
        """
        Get the original source file if it exists.

        Override this method to specify where the original file is located.
        Different books may have different naming conventions for the original file.

        Returns:
            Path to the original file, or None if not available
        """
        # Look for files starting with "00-" in the output directory
        if not self.config.source_output_dir.exists():
            return None

        for file_path in self.config.source_output_dir.glob("00-*"):
            if file_path.is_file():
                return file_path

        return None


def build(version: str, name: str):
    """
    Build function for this specific book.

    Args:
        version: Version string for the build
        name: Name of the book directory
    """
    # Create book configuration
    config = BookConfig(
        name=name,
        version=version,
        title="[Your Book Title Here]",
        author="[Your Book Author Here]",
        # Optional: specify a custom clean title for filenames
        # clean_title="Your-Custom-Clean-Title"
    )

    # Create and run the builder
    builder = YourBookBuilder(config=config)
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
