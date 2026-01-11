"""
Base builder class for creating consistent book builds across different projects.

This module provides a shared foundation for building books from markdown sources
to various output formats (EPUB, PDF) while maintaining consistent naming and
file organization patterns.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import pypandoc  # type: ignore[import-untyped]
from loguru import logger
from PIL import Image
from PIL.Image import Image as PilImage


class BookConfig:
    """
    Configuration class for book-specific settings.

    This class holds all the metadata and file paths specific to a book,
    ensuring consistent naming conventions across all builds.
    """

    def __init__(
        self,
        name: str,
        version: str,
        title: str,
        author: str,
        clean_title: Optional[str] = None,
    ):
        """
        Initialize book configuration.

        Args:
            name: Directory name for the book (e.g., "the_federalist_papers")
            version: Version string (e.g., "1.0.0")
            title: Full book title for metadata
            author: Book author(s) for metadata
            clean_title: Clean version of title for filenames (auto-generated if None)
        """
        if name is None:
            raise ValueError("Book name cannot be None")

        self.name = name
        self.version = version
        self.title = title
        self.author = author
        self.clean_title = clean_title or self._generate_clean_title(title)

        # Source Paths
        self.base_dir = Path(f"books/{self.name}")
        self.source_output_dir = self.base_dir / "output"

        # Intermediate (Staging) Paths
        self.staging_dir = self.base_dir / "staging"
        self.staged_modernized_md = self.staging_dir / "modernized.md"
        self.staged_annotated_md = self.staging_dir / "annotated.md"
        self.staged_metadata_json = self.staging_dir / "metadata.json"
        self.staged_original_md = self.staging_dir / "original.md"

        # Build Output Paths
        self.build_dir = Path("build") / self.name / self.version
        self.build_modernized_md = self.build_dir / f"{self.clean_title}-modernized.md"
        self.build_annotated_md = self.build_dir / f"{self.clean_title}-annotated.md"
        self.build_original_md = self.build_dir / f"{self.clean_title}-original.md"
        self.build_metadata_json = self.build_dir / f"{self.clean_title}-metadata.json"
        self.build_modernized_epub = self.build_dir / f"{self.clean_title}-modernized.epub"
        self.build_modernized_pdf = self.build_dir / f"{self.clean_title}-modernized.pdf"
        self.build_annotated_epub = self.build_dir / f"{self.clean_title}-annotated.epub"
        self.build_annotated_pdf = self.build_dir / f"{self.clean_title}-annotated.pdf"

        # Asset Paths
        self.cover_image = self._get_cover_image()
        self.epub_css = self._get_asset_path("epub.css")
        self.preface_md = self._get_asset_path("preface.md")
        self.license_md = self._get_asset_path("license.md")

        # Base Directory Output Paths
        self.base_modernized_md = self.base_dir / "output-modernized.md"
        self.base_annotated_md = self.base_dir / "output-annotated.md"
        self.base_original_md = self.base_dir / "output-original.md"
        self.base_metadata_json = self.base_dir / "metadata.json"

    def _generate_clean_title(self, title: str) -> str:
        """
        Generate a clean title suitable for filenames.

        Args:
            title: The original title

        Returns:
            Clean title with special characters replaced
        """
        # Replace special characters and spaces with hyphens
        clean = re.sub(r"[^\w\s-]", "", title)
        clean = re.sub(r"[-\s_]+", "-", clean)
        return clean.strip("-")

    def _get_cover_image(self) -> Path:
        """
        Get the path for the cover image, supporting both PNG and WebP formats.
        Checks book-specific directory first, then falls back to default render assets.

        Returns:
            Path to the cover image file (PNG or WebP)

        Raises:
            FileNotFoundError: If no cover image is found in either format
        """
        # Check for book-specific cover in both formats
        for extension in ["webp", "png"]:
            book_specific_path = self.base_dir / f"cover.{extension}"
            if book_specific_path.exists():
                return book_specific_path

        # Check for default cover in both formats
        for extension in ["webp", "png"]:
            default_path = Path("books/default_render_assets") / f"cover.{extension}"
            if default_path.exists():
                return default_path

        # If no cover found, return a default path (will be checked later)
        return self.base_dir / "cover.webp"

    def _get_asset_path(self, filename: str) -> Path:
        """
        Get the path for an asset file, checking book-specific directory first,
        then falling back to default render assets.

        Args:
            filename: The name of the asset file

        Returns:
            Path to the asset file (book-specific or default)
        """
        book_specific_path = self.base_dir / filename
        if book_specific_path.exists():
            return book_specific_path

        default_path = Path("books/default_render_assets") / filename
        return default_path

    def get_source_files(self) -> Dict[str, Path]:
        """
        Get the source file paths for this book.

        Returns:
            Dictionary mapping file types to their source paths
        """
        return {
            "modernized": self.source_output_dir / "03-input_small Final_1.md",
            "annotated": self.source_output_dir / "06-input_small Annotate_1.md",
        }


class BaseBookBuilder(ABC):
    """
    Abstract base class for building books.

    This class provides all the common build logic while allowing subclasses
    to customize the build process for their specific needs.
    """

    def __init__(self, config: BookConfig):
        """
        Initialize the builder with a book configuration.

        Args:
            config: Book configuration object
        """
        self.config = config

    @abstractmethod
    def get_source_files(self) -> Dict[str, Path]:
        """
        Get the source file paths for this specific book.

        This method should be implemented by subclasses to specify
        where their source files are located, as the naming may vary.

        Returns:
            Dictionary mapping file types to their source paths
        """
        pass

    @abstractmethod
    def get_original_file(self) -> Optional[Path]:
        """
        Get the original source file if it exists.

        Returns:
            Path to the original file, or None if not available
        """
        pass

    def safe_relative_path(self, path: Path) -> str:
        """
        Safely gets a relative path for logging, falling back to absolute path if needed.

        Args:
            path: The path to convert to relative format

        Returns:
            A string representation of the path, relative if possible, absolute otherwise
        """
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return str(path)

    def ensure_directories_exist(self) -> None:
        """
        Ensures all necessary directories exist for the build process.
        """
        directories_to_create = [
            self.config.base_dir,
            self.config.build_dir,
            self.config.staging_dir,
            self.config.source_output_dir,
        ]

        for directory in directories_to_create:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {self.safe_relative_path(directory)}")

        # Verify that source files exist
        source_files = self.get_source_files()
        for file_type, file_path in source_files.items():
            if not file_path.exists():
                raise FileNotFoundError(
                    f"{file_type.capitalize()} book file not found: {self.safe_relative_path(file_path)}"
                )

    def get_latest_metadata_file(self) -> Optional[Path]:
        """
        Finds the latest (by filename) metadata file.

        Returns:
            Path to the latest metadata file, or None if not found
        """
        files = sorted(self.config.source_output_dir.glob("pipeline_metadata*.json"), reverse=True)
        return files[0] if files else None

    def replace_br_tags(self, input_path: Path) -> None:
        """
        Replaces '<br>' tags with two spaces in a markdown file.

        Args:
            input_path: Path to the markdown file to process
        """
        content = input_path.read_text(encoding="utf-8")
        content = content.replace("<br>", "  ")
        input_path.write_text(content, encoding="utf-8")
        logger.info(f"Replaced <br> tags with spaces in '{self.safe_relative_path(input_path)}'")

    def clean_annotation_patterns(self, input_path: Path) -> None:
        """
        Removes specific annotation end patterns and cleans up resulting empty lines.

        Removes the following patterns (case insensitive):
        - **End annotation.**
        - **End annotated introduction.**
        - **End annotated summary.**
        - **End quote.**

        If removal results in an empty line or a line with only '>', deletes that line.

        Args:
            input_path: Path to the markdown file to process
        """
        content = input_path.read_text(encoding="utf-8")

        # Patterns to remove (case insensitive)
        patterns_to_remove = [
            r"\*\*End annotation\.\*\*",
            r"\*\*End annotated introduction\.\*\*",
            r"\*\*End annotated summary\.\*\*",
            r"\*\*End quote\.\*\*",
        ]

        # Split into lines to track which ones are affected
        lines = content.split("\n")
        affected_lines = set()

        # Process each line and track which ones are affected by substitutions
        for i, line in enumerate(lines):
            original_line = line
            modified_line = line

            # Apply each pattern substitution
            for pattern in patterns_to_remove:
                modified_line = re.sub(pattern, "", modified_line, flags=re.IGNORECASE)

            # If the line was modified, mark it as affected
            if modified_line != original_line:
                affected_lines.add(i)
                lines[i] = modified_line

        # Now filter out empty lines or lines with only '>' only if they were affected
        cleaned_lines = []
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            # Keep the line if it wasn't affected, or if it's not empty and not just '>'
            if i not in affected_lines or (stripped_line and stripped_line != ">"):
                cleaned_lines.append(line)

        # Rejoin the lines
        cleaned_content = "\n".join(cleaned_lines)

        input_path.write_text(cleaned_content, encoding="utf-8")
        logger.info(f"Cleaned annotation patterns from '{self.safe_relative_path(input_path)}'")

    def format_markdown_file(self, input_path: Path, preface_content: str, license_content: str, version: str) -> None:
        """
        Reads a Markdown file, replaces placeholders, and writes it back.

        Args:
            input_path: Path to the markdown file to format
            preface_content: Content to replace {preface} placeholder
            license_content: Content to replace {license} placeholder
            version: Version string to replace {version} placeholder
        """
        content = input_path.read_text(encoding="utf-8")
        # Use specific placeholder replacement to avoid issues with braces in content
        content = content.replace("{preface}", preface_content)
        content = content.replace("{license}", license_content)
        content = content.replace("{version}", version)
        input_path.write_text(content, encoding="utf-8")
        logger.info(f"Formatted '{self.safe_relative_path(input_path)}' with preface, license, and version.")

    def copy_and_prepare_files(self) -> None:
        """
        Copy source files to staging area and prepare them for building.
        """
        logger.info("Preparing Source Files")

        # Copy main source files
        source_files = self.get_source_files()
        for file_type, source_path in source_files.items():
            if file_type == "modernized":
                dest_path = self.config.staged_modernized_md
            elif file_type == "annotated":
                dest_path = self.config.staged_annotated_md
            else:
                continue

            shutil.copy(source_path, dest_path)
            logger.info(f"Copied '{self.safe_relative_path(source_path)}' to staging area.")

        # Copy original file to staging if it exists
        original_file = self.get_original_file()
        if original_file and original_file.exists():
            shutil.copy(original_file, self.config.staged_original_md)
            logger.info(f"Copied '{self.safe_relative_path(original_file)}' to staging area.")
        else:
            logger.warning(
                f"Original file not found at '{self.safe_relative_path(original_file) if original_file else 'None'}'"
            )

        # Copy metadata
        latest_metadata = self.get_latest_metadata_file()
        if latest_metadata:
            shutil.copy(latest_metadata, self.config.staged_metadata_json)
            logger.info(f"Copied latest metadata '{self.safe_relative_path(latest_metadata)}' to staging.")
        else:
            logger.warning("No metadata file found. Build will proceed with defaults.")

    def format_markdown_files(self) -> None:
        """
        Format all markdown files with preface, license, and version information.
        """
        logger.info("Formatting Markdown")

        # Check required files exist (either book-specific or default)
        if not self.config.preface_md.exists():
            raise FileNotFoundError(f"Preface file not found: {self.safe_relative_path(self.config.preface_md)}")
        if not self.config.license_md.exists():
            raise FileNotFoundError(f"License file not found: {self.safe_relative_path(self.config.license_md)}")

        preface_content = self.config.preface_md.read_text(encoding="utf-8")
        license_content = self.config.license_md.read_text(encoding="utf-8")

        # Format license content with title and author
        license_content = license_content.replace("{title}", self.config.title)
        license_content = license_content.replace("{author}", self.config.author)

        # Format main files
        self.format_markdown_file(
            self.config.staged_modernized_md, preface_content, license_content, self.config.version
        )
        self.format_markdown_file(
            self.config.staged_annotated_md, preface_content, license_content, self.config.version
        )

        # Format original file if it exists
        if self.config.staged_original_md.exists():
            self.format_markdown_file(
                self.config.staged_original_md, preface_content, license_content, self.config.version
            )

    def clean_markdown_files(self) -> None:
        """
        Clean markdown files by replacing <br> tags, cleaning annotation patterns,
        and replacing pipeline-specific markers with reader-friendly alternatives.
        """
        logger.info("Replacing <br> tags with spaces")
        self.replace_br_tags(self.config.staged_modernized_md)
        self.replace_br_tags(self.config.staged_annotated_md)
        if self.config.staged_original_md.exists():
            self.replace_br_tags(self.config.staged_original_md)

        logger.info("Cleaning annotation patterns")
        self.clean_annotation_patterns(self.config.staged_modernized_md)
        self.clean_annotation_patterns(self.config.staged_annotated_md)
        if self.config.staged_original_md.exists():
            self.clean_annotation_patterns(self.config.staged_original_md)

        logger.info("Replacing pipeline-specific markers with reader-friendly alternatives")
        self.clean_start_markers(self.config.staged_modernized_md)
        self.clean_start_markers(self.config.staged_annotated_md)
        if self.config.staged_original_md.exists():
            self.clean_start_markers(self.config.staged_original_md)

        logger.info("Loosening blockquote lists for EPUB/PDF rendering")
        self.loosen_blockquote_lists(self.config.staged_modernized_md)
        self.loosen_blockquote_lists(self.config.staged_annotated_md)
        if self.config.staged_original_md.exists():
            self.loosen_blockquote_lists(self.config.staged_original_md)

    @staticmethod
    def make_blockquote_lists_loose(markdown_text: str) -> str:
        """Convert tight blockquote "list-like" blocks into loose blocks by inserting blank quote lines.

        Pandoc/EPUB rendering often collapses visual spacing for "tight" Markdown lists,
        especially when lists appear inside blockquotes.

        This function targets two common patterns:
        1) True Markdown list items inside blockquotes (e.g., `> - item`, `> * item`, `> 1. item`)
        2) "Term-definition" lines commonly emitted by the LLM (e.g., `> *Akbar* - ...`)

        It inserts `>` blank lines before these entries when the previous line is also
        part of a blockquote but not already separated by a blank quote line.

        Args:
            markdown_text: Raw Markdown content.

        Returns:
            Updated Markdown content with loose blockquote lists.
        """
        blockquote_line_re = re.compile(r"^\s*>")
        blockquote_list_item_start_re = re.compile(r"^\s*>\s*(?:[*+-]|\d+[.)])\s+")
        blockquote_term_definition_re = re.compile(r"^\s*>\s*\*[^*].*?\*\s*-\s+")

        def is_blockquote_line(line: str) -> bool:
            return bool(blockquote_line_re.match(line))

        def is_blank_blockquote_line(line: str) -> bool:
            return line.strip() == ">"

        def is_blockquote_list_item_start(line: str) -> bool:
            return bool(blockquote_list_item_start_re.match(line))

        def is_blockquote_term_definition(line: str) -> bool:
            return bool(blockquote_term_definition_re.match(line))

        def is_blockquote_entry(line: str) -> bool:
            return is_blockquote_list_item_start(line) or is_blockquote_term_definition(line)

        lines = markdown_text.splitlines()
        out_lines: list[str] = []

        for line in lines:
            if is_blockquote_entry(line) and out_lines:
                prev_line = out_lines[-1]
                if is_blockquote_line(prev_line) and not is_blank_blockquote_line(prev_line) and prev_line.strip():
                    out_lines.append(">")
            out_lines.append(line)

        out_text = "\n".join(out_lines)
        if markdown_text.endswith("\n"):
            out_text += "\n"
        return out_text

    def loosen_blockquote_lists(self, input_path: Path) -> None:
        """Rewrite tight blockquote lists to loose lists to improve EPUB/PDF spacing.

        Args:
            input_path: Path to the markdown file to process.
        """
        content = input_path.read_text(encoding="utf-8")
        updated = self.make_blockquote_lists_loose(content)
        if updated == content:
            return

        input_path.write_text(updated, encoding="utf-8")
        logger.info(f"Loosened blockquote lists in '{self.safe_relative_path(input_path)}'")

    def clean_start_markers(self, input_path: Path) -> None:
        """Replace pipeline-specific blockquote markers with reader-friendly alternatives.

        Transforms the following markers within blockquotes:
        - **Annotation:** -> **Note:**
        - **Annotated introduction:** -> **Overview:**
        - **Annotated summary:** -> **Summary:**
        - **Quote:** is left unchanged (already reader-friendly)

        Args:
            input_path: Path to the markdown file to process.
        """
        content = input_path.read_text(encoding="utf-8")

        # Define marker replacements (pattern -> replacement)
        # Note: These patterns should only match within blockquotes (lines starting with >)
        replacements = [
            (r"(>\s*)\*\*Annotation:\*\*", r"\1**Note:**"),
            (r"(>\s*)\*\*Annotated introduction:\*\*", r"\1**Introduction:**"),
            (r"(>\s*)\*\*Annotated summary:\*\*", r"\1**Summary:**"),
        ]

        modified = False
        for pattern, replacement in replacements:
            new_content = re.sub(pattern=pattern, repl=replacement, string=content, flags=re.MULTILINE)
            if new_content != content:
                modified = True
                content = new_content

        if modified:
            input_path.write_text(content, encoding="utf-8")
            logger.info(f"Cleaned start markers in '{self.safe_relative_path(input_path)}'")

    def _ensure_compatible_cover_image(self) -> Optional[Path]:
        """
        Ensures the cover image is in a compatible format (JPEG) for PDF conversion.
        Converts PNG/WebP to JPEG if necessary.

        Returns:
            Path to the compatible cover image, or None if no cover exists
        """
        if not self.config.cover_image.exists():
            return None

        # Check if it's already a JPEG
        if self.config.cover_image.suffix.lower() in [".jpg", ".jpeg"]:
            return self.config.cover_image

        # Convert to JPEG in staging directory
        converted_path = self.config.staging_dir / "cover.jpg"

        try:
            logger.info("Converting cover image to JPEG for compatibility...")
            img: PilImage = Image.open(self.config.cover_image)

            # Convert RGBA to RGB if necessary (for PNG with transparency)
            if img.mode in ("RGBA", "LA", "P"):
                # Create a white background
                background: PilImage = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Save as JPEG with good quality
            img.save(converted_path, "JPEG", quality=95)
            logger.success(f"Converted cover image to JPEG: {self.safe_relative_path(converted_path)}")
            return converted_path
        except Exception as e:
            logger.error(f"Failed to convert cover image: {e}")
            return None

    def build_epub_and_pdf(self) -> None:
        """
        Build EPUB and PDF files using Pandoc and Calibre.
        """
        logger.info("Running Pandoc Build")

        # Update metadata with version
        metadata = {}
        if self.config.staged_metadata_json.exists():
            with open(self.config.staged_metadata_json, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                metadata["book_version"] = self.config.version
            with open(self.config.staged_metadata_json, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)

        pandoc_args = [
            "--toc",
            "--toc-depth=3",
            f"--css={self.safe_relative_path(self.config.epub_css)}",
            "--lua-filter=books/exclude_h1.lua",
        ]

        # Ensure we have a compatible cover image
        compatible_cover = self._ensure_compatible_cover_image()

        # Build Modernized Version
        logger.info(f"Building modernized version for '{self.config.title}'...")
        cover_args = []
        if compatible_cover:
            cover_args.append(f"--epub-cover-image={self.safe_relative_path(compatible_cover)}")
        else:
            logger.warning("No cover image available for EPUB")

        pypandoc.convert_file(
            str(self.config.staged_modernized_md),
            "epub2",
            outputfile=str(self.config.build_modernized_epub),
            extra_args=pandoc_args
            + cover_args
            + [
                f'--metadata=title:"{self.config.title}"',
                f'--metadata=author:"{self.config.author}"',
                f'--metadata=version:"{metadata.get("book_version", self.config.version)}"',
            ],
        )
        logger.success(f"  -> Created '{self.safe_relative_path(self.config.build_modernized_epub)}'")

        # Build PDF from EPUB for better formatting
        logger.info("Building PDF from EPUB for modernized version...")
        success = self._build_pdf_from_epub(
            epub_path=self.config.build_modernized_epub,
            pdf_path=self.config.build_modernized_pdf,
            title=self.config.title,
            author=self.config.author,
            version=metadata.get("book_version", self.config.version),
        )
        if success:
            logger.success(f"  -> Created '{self.safe_relative_path(self.config.build_modernized_pdf)}'")
        else:
            logger.error(f"  -> Failed to create '{self.safe_relative_path(self.config.build_modernized_pdf)}'")

        # Build Annotated Version
        logger.info(f"Building annotated version for '{self.config.title}'...")
        pypandoc.convert_file(
            str(self.config.staged_annotated_md),
            "epub2",
            outputfile=str(self.config.build_annotated_epub),
            extra_args=pandoc_args
            + cover_args
            + [
                f'--metadata=title:"{self.config.title}"',
                f'--metadata=author:"{self.config.author}"',
                f'--metadata=version:"{metadata.get("book_version", self.config.version)}"',
            ],
        )
        logger.success(f"  -> Created '{self.safe_relative_path(self.config.build_annotated_epub)}'")

        # Build PDF from EPUB for better formatting
        logger.info("Building PDF from EPUB for annotated version...")
        success = self._build_pdf_from_epub(
            epub_path=self.config.build_annotated_epub,
            pdf_path=self.config.build_annotated_pdf,
            title=f"{self.config.title}",
            author=f"{self.config.author}",
            version=metadata.get("book_version", self.config.version),
        )
        if success:
            logger.success(f"  -> Created '{self.safe_relative_path(self.config.build_annotated_pdf)}'")
        else:
            logger.error(f"  -> Failed to create '{self.safe_relative_path(self.config.build_annotated_pdf)}'")

    def _build_pdf_from_epub(self, epub_path: Path, pdf_path: Path, title: str, author: str, version: str) -> bool:
        """
        Builds PDF from EPUB using Calibre's ebook-convert for perfect conversion.

        Args:
            epub_path: Path to the source EPUB file
            pdf_path: Path for the output PDF file
            title: Book title for metadata
            author: Book author for metadata
            version: Book version for metadata

        Returns:
            True if PDF was successfully created, False otherwise
        """
        try:
            logger.info("Converting EPUB to PDF using Calibre ebook-convert...")

            # Set environment variable to disable GUI components that might cause issues
            env = os.environ.copy()
            env["QT_QPA_PLATFORM"] = "offscreen"

            # Use Calibre's ebook-convert with minimal options to avoid image format issues
            result = subprocess.run(
                [
                    "ebook-convert",
                    str(epub_path),
                    str(pdf_path),
                    "--pdf-default-font-size",
                    "12",
                    "--pdf-mono-font-size",
                    "10",
                    "--paper-size",
                    "letter",
                    "--pdf-page-margin-left",
                    "72",
                    "--pdf-page-margin-right",
                    "72",
                    "--pdf-page-margin-top",
                    "72",
                    "--pdf-page-margin-bottom",
                    "72",
                ],
                capture_output=True,
                text=True,
                env=env,
                check=False,  # Don't raise on non-zero exit
            )

            if result.returncode == 0:
                logger.success("Successfully created PDF using Calibre")
                return True
            else:
                # Check if the PDF was actually created despite the error
                if pdf_path.exists() and pdf_path.stat().st_size > 0:
                    logger.warning(f"PDF created with warnings: {result.stderr}")
                    return True
                else:
                    logger.error(f"Calibre conversion failed: {result.stderr}")
                    return False

        except FileNotFoundError:
            logger.error(
                "Calibre ebook-convert not found. Please install Calibre and ensure ebook-convert is in your PATH."
            )
            return False
        except Exception as e:
            logger.error(f"Failed to create PDF: {str(e)}")
            return False

    def copy_final_artifacts(self) -> None:
        """
        Copy final artifacts to build directory and clean up staging.
        """
        logger.info("Copying Final Artifacts to Build Directory")

        # Copy markdown files
        shutil.copy(self.config.staged_modernized_md, self.config.build_modernized_md)
        logger.info(f"Copied modernized markdown to '{self.safe_relative_path(self.config.build_modernized_md)}'")

        shutil.copy(self.config.staged_annotated_md, self.config.build_annotated_md)
        logger.info(f"Copied annotated markdown to '{self.safe_relative_path(self.config.build_annotated_md)}'")

        if self.config.staged_original_md.exists():
            shutil.copy(self.config.staged_original_md, self.config.build_original_md)
            logger.info(f"Copied original manuscript to '{self.safe_relative_path(self.config.build_original_md)}'")
        else:
            logger.warning("Original file not found in staging area")

        # Copy metadata
        if self.config.staged_metadata_json.exists():
            shutil.copy(self.config.staged_metadata_json, self.config.build_metadata_json)
            logger.info(f"Copied metadata to '{self.safe_relative_path(self.config.build_metadata_json)}'")

        # Clean up staging directory
        shutil.rmtree(self.config.staging_dir)
        logger.info(f"Cleaned up staging directory: '{self.safe_relative_path(self.config.staging_dir)}'")

    def build(self) -> None:
        """
        Execute the complete build process.
        """
        logger.info(f"Starting build for '{self.config.name}' version '{self.config.version}'")

        # Setup Directories
        self.ensure_directories_exist()
        logger.info(f"Build output directory: '{self.safe_relative_path(self.config.build_dir)}'")

        # Copy and Prepare Files
        self.copy_and_prepare_files()

        # Format Markdown
        self.format_markdown_files()

        # Clean Markdown
        self.clean_markdown_files()

        # Build EPUB and PDF
        self.build_epub_and_pdf()

        # Copy Final Artifacts
        self.copy_final_artifacts()

        logger.success(f"\nBuild complete for {self.config.name} {self.config.version}!")


def create_build_parser() -> argparse.ArgumentParser:
    """
    Create a standardized argument parser for build scripts.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Build script for Markdown to EPUB/PDF.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the book.")
    build_parser.add_argument("version", type=str, help="The version string for the build (e.g., '1.0.0').")
    build_parser.add_argument(
        "--name",
        type=str,
        help="The name of the book, used for directory paths. Auto-detected from module path if not provided.",
    )

    return parser


def auto_detect_book_name() -> str:
    """
    Auto-detect the book name from the calling module path.

    Detection strategy (in order):
    1. Direct script execution: Extract from path like books/{book_name}/build.py
    2. Module execution: Extract from module path like books.{book_name}.build
    3. Current working directory: Extract from CWD if it contains books/{book_name}

    Returns:
        The book name extracted from the module path

    Raises:
        ValueError: If book name cannot be detected from any strategy
    """
    import sys

    # Strategy 1: Direct script execution
    if len(sys.argv) > 0 and sys.argv[0].endswith(".py"):
        name = _extract_from_script_path(sys.argv[0])
        if name:
            return name

    # Strategy 2: Module execution
    if len(sys.argv) > 1 and sys.argv[0] == "-m":
        name = _extract_from_module_path(sys.argv[1])
        if name:
            return name

    # Strategy 3: Current working directory
    name = _extract_from_cwd()
    if name:
        return name

    raise ValueError("Could not auto-detect book name. Please provide --name argument.")


def _extract_from_script_path(script_path: str) -> Optional[str]:
    """
    Extract book name from script path.

    Args:
        script_path: Path to the script file

    Returns:
        Book name if found, None otherwise

    Examples:
        >>> _extract_from_script_path("books/the_federalist_papers/build.py")
        "the_federalist_papers"
        >>> _extract_from_script_path("/path/to/books/example_book/build.py")
        "example_book"
    """
    path_parts = script_path.replace("\\", "/").split("/")
    if "books" in path_parts:
        books_index = path_parts.index("books")
        if books_index + 1 < len(path_parts):
            return path_parts[books_index + 1]
    return None


def _extract_from_module_path(module_path: str) -> Optional[str]:
    """
    Extract book name from module path.

    Args:
        module_path: Module path string

    Returns:
        Book name if found, None otherwise

    Examples:
        >>> _extract_from_module_path("books.the_federalist_papers.build")
        "the_federalist_papers"
        >>> _extract_from_module_path("books.example_book.build")
        "example_book"
    """
    if module_path.startswith("books."):
        parts = module_path.split(".")
        if len(parts) >= 2:
            return parts[1]  # books.the_federalist_papers.build -> the_federalist_papers
    return None


def _extract_from_cwd() -> Optional[str]:
    """
    Extract book name from current working directory.

    Returns:
        Book name if found, None otherwise

    Examples:
        >>> # When CWD is /path/to/books/the_federalist_papers
        >>> _extract_from_cwd()
        "the_federalist_papers"
    """
    import os

    cwd = os.getcwd()
    cwd_parts = cwd.replace("\\", "/").split("/")
    if "books" in cwd_parts:
        books_index = cwd_parts.index("books")
        if books_index + 1 < len(cwd_parts):
            return cwd_parts[books_index + 1]
    return None
