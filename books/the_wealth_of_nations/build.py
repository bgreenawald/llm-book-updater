"""
Build script for The Wealth of Nations (Timeless Library Edition).

This script uses the shared BaseBookBuilder to create consistent builds
while allowing for book-specific customization.
"""

from pathlib import Path
from typing import Dict, Optional

from books.base_builder import (
    BaseBookBuilder,
    BookConfig,
    auto_detect_book_name,
    create_build_parser,
)


class WealthOfNationsBuilder(BaseBookBuilder):
    """
    Book-specific builder for The Wealth of Nations.

    This class implements the abstract methods required by BaseBookBuilder
    to specify where the source files are located for this particular book.
    """

    def get_source_files(self) -> Dict[str, Path]:
        """
        Get the source file paths for The Wealth of Nations.

        Returns:
            Dictionary mapping file types to their source paths
        """
        return {
            "modernized": self.config.source_output_dir / "03-input_transformed Final_two_stage_1.md",
            "annotated": self.config.source_output_dir / "06-input_transformed Annotate_1.md",
        }

    def get_original_file(self) -> Optional[Path]:
        """
        Get the original source file for The Wealth of Nations.

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
    Build function for The Wealth of Nations.

    Args:
        version: Version string for the build
        name: Name of the book directory
    """
    # Create book configuration
    config = BookConfig(
        name=name,
        version=version,
        title="The Wealth of Nations (Timeless Library Edition)",
        author="Adam Smith",
    )

    # Create and run the builder
    builder = WealthOfNationsBuilder(config=config)
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
