"""Build script for The Essence of Christianity (Timeless Library Edition).

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


class EssenceOfChristianityBuilder(BaseBookBuilder):
    """Book-specific builder for The Essence of Christianity.

    This class implements the abstract methods required by BaseBookBuilder
    to specify where the source files are located for this particular book.
    """

    def get_source_files(self) -> Dict[str, Path]:
        """Get the source file paths for The Essence of Christianity.

        Returns:
            Dictionary mapping file types to their source paths.

        Raises:
            FileNotFoundError: If required source files are missing.
        """
        source_files = {
            "modernized": self.config.source_output_dir / "03-input_transformed Final_1.md",
            "annotated": self.config.source_output_dir / "06-input_transformed Annotate_1.md",
        }

        for file_type, file_path in source_files.items():
            if not file_path.exists():
                raise FileNotFoundError(f"Required {file_type} source file not found: {file_path}")

        return source_files

    def get_original_file(self) -> Optional[Path]:
        """Get the original source file for The Essence of Christianity.

        Returns:
            Path to the original file, or None if not available.
        """
        if not self.config.source_output_dir.exists():
            return None

        for file_path in self.config.source_output_dir.glob("00-*"):
            if file_path.is_file():
                return file_path

        return None


def build(version: str, name: str) -> None:
    """Build The Essence of Christianity.

    Args:
        version: Version string for the build.
        name: Name of the book directory.
    """
    config = BookConfig(
        name=name,
        version=version,
        title="The Essence of Christianity (Timeless Library Edition)",
        author="Ludwig Feuerbach",
    )

    builder = EssenceOfChristianityBuilder(config=config)
    builder.build()


def main() -> None:
    """Parse CLI args and run the build."""
    parser = create_build_parser()
    args = parser.parse_args()

    if args.command == "build":
        book_name = args.name or auto_detect_book_name()
        build(version=args.version, name=book_name)


if __name__ == "__main__":
    main()
