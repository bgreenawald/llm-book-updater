import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import pypandoc
from loguru import logger


class Config:
    """
    Configuration class for the build script.
    Manages all the necessary file paths.
    """

    BOOK_TITLE = "The Federalist Papers (Modern AI Edition)"
    BOOK_AUTHOR = "Alexander Hamilton, James Madison, and John Jay"

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version

        # --- Source Paths ---
        self.base_dir = Path(f"books/{self.name}")
        self.source_output_dir = self.base_dir / "output"
        self.final_book_path = self.source_output_dir / "03-input_small Final_1.md"
        self.annotated_book_path = self.source_output_dir / "06-input_small Annotate_1.md"
        self.original_file_path = self.base_dir / "input_small.md"

        # --- Intermediate (Staging) Paths ---
        self.staging_dir = self.base_dir / "staging"
        self.staged_final_md = self.staging_dir / "final.md"
        self.staged_annotated_md = self.staging_dir / "annotated.md"
        self.staged_metadata_json = self.staging_dir / "metadata.json"

        # --- Build Output Paths ---
        self.build_dir = Path("build") / self.name / self.version
        self.build_final_md = self.build_dir / f"{self.BOOK_TITLE}-final.md"
        self.build_annotated_md = self.build_dir / f"{self.BOOK_TITLE}-annotated.md"
        self.build_original_md = self.build_dir / f"{self.BOOK_TITLE}-original.md"
        self.build_metadata_json = self.build_dir / f"{self.BOOK_TITLE}-metadata.json"
        self.build_final_epub = self.build_dir / f"{self.BOOK_TITLE}-final.epub"
        self.build_final_pdf = self.build_dir / f"{self.BOOK_TITLE}-final.pdf"
        self.build_annotated_epub = self.build_dir / f"{self.BOOK_TITLE}-annotated.epub"
        self.build_annotated_pdf = self.build_dir / f"{self.BOOK_TITLE}-annotated.pdf"

        # --- Asset Paths ---
        self.cover_image = self.base_dir / "cover.png"
        self.epub_css = self.base_dir / "epub.css"
        self.preface_md = self.base_dir / "preface.md"
        self.license_md = self.base_dir / "license.md"

        # --- Base Directory Output Paths ---
        self.base_final_md = self.base_dir / "output-final.md"
        self.base_annotated_md = self.base_dir / "output-annotated.md"


def safe_relative_path(path: Path) -> str:
    """
    Safely gets a relative path for logging, falling back to absolute path if needed.

    Args:
        path: The path to convert to relative format.

    Returns:
        A string representation of the path, relative if possible, absolute otherwise.
    """
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def ensure_directories_exist(config: Config) -> None:
    """
    Ensures all necessary directories exist for the build process.

    Args:
        config: The configuration object containing all file paths.
    """
    directories_to_create = [
        config.base_dir,
        config.build_dir,
        config.staging_dir,
        config.source_output_dir,
    ]

    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {safe_relative_path(directory)}")

    # Verify that source files exist
    if not config.final_book_path.exists():
        raise FileNotFoundError(f"Final book file not found: {safe_relative_path(config.final_book_path)}")
    if not config.annotated_book_path.exists():
        raise FileNotFoundError(f"Annotated book file not found: {safe_relative_path(config.annotated_book_path)}")


def get_latest_metadata_file(source_output_dir: Path) -> Path | None:
    """
    Finds the latest (by filename) metadata file.
    """
    files = sorted(source_output_dir.glob("pipeline_metadata*.json"), reverse=True)
    return files[0] if files else None


def replace_br_tags(input_path: Path) -> None:
    """
    Replaces '<br>' tags with two spaces in a markdown file.

    Args:
        input_path: Path to the markdown file to process.
    """
    content = input_path.read_text(encoding="utf-8")
    content = content.replace("<br>", "  ")
    input_path.write_text(content, encoding="utf-8")
    logger.info(f"Replaced <br> tags with spaces in '{safe_relative_path(input_path)}'")


def clean_annotation_patterns(input_path: Path) -> None:
    """
    Removes specific annotation end patterns and cleans up resulting empty lines.

    Removes the following patterns (case insensitive):
    - **End annotation.**
    - **End annotated introduction.**
    - **End annotated summary.**
    - **End quote.**

    If removal results in an empty line or a line with only '>', deletes that line.

    Args:
        input_path: Path to the markdown file to process.
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
    logger.info(f"Cleaned annotation patterns from '{safe_relative_path(input_path)}'")


 def format_markdown_file(input_path: Path, preface_content: str, license_content: str):
     """
     Reads a Markdown file, replaces placeholders, and writes it back.
     """
     content = input_path.read_text(encoding="utf-8")
-    content = content.format(preface=preface_content, license=license_content)
+    # Use specific placeholder replacement to avoid issues with braces in content
+    content = content.replace("{preface}", preface_content)
+    content = content.replace("{license}", license_content)
     input_path.write_text(content, encoding="utf-8")
     logger.info(f"Formatted '{safe_relative_path(input_path)}' with preface and license.")

def build(version: str, name: str):
    """
    The main build function.
    """
    config = Config(name=name, version=version)

    # --- 1. Setup Directories ---
    logger.info(f"Starting build for '{name}' version '{version}'")
    ensure_directories_exist(config)
    logger.info(f"Build output directory: '{safe_relative_path(config.build_dir)}'")

    # --- 2. Copy and Prepare Files ---
    logger.info("--- Preparing Source Files ---")
    shutil.copy(config.final_book_path, config.staged_final_md)
    logger.info(f"Copied '{safe_relative_path(config.final_book_path)}' to staging area.")
    shutil.copy(config.annotated_book_path, config.staged_annotated_md)
    logger.info(f"Copied '{safe_relative_path(config.annotated_book_path)}' to staging area.")

    latest_metadata = get_latest_metadata_file(config.source_output_dir)
    if latest_metadata:
        shutil.copy(latest_metadata, config.staged_metadata_json)
        logger.info(f"Copied latest metadata '{safe_relative_path(latest_metadata)}' to staging.")
    else:
        logger.warning("No metadata file found. Build will proceed with defaults.")

    # --- 3. Format Markdown ---
    logger.info("--- Formatting Markdown ---")
    if not config.preface_md.exists():
        raise FileNotFoundError(f"Preface file not found: {safe_relative_path(config.preface_md)}")
    if not config.license_md.exists():
        raise FileNotFoundError(f"License file not found: {safe_relative_path(config.license_md)}")
    preface_content = config.preface_md.read_text(encoding="utf-8")
    license_content = config.license_md.read_text(encoding="utf-8")

    format_markdown_file(config.staged_final_md, preface_content, license_content)
    format_markdown_file(config.staged_annotated_md, preface_content, license_content)

    # --- 3.5. Replace <br> tags with spaces ---
    logger.info("--- Replacing <br> tags with spaces ---")
    replace_br_tags(config.staged_final_md)
    replace_br_tags(config.staged_annotated_md)

    # --- 3.6. Clean annotation patterns ---
    logger.info("--- Cleaning annotation patterns ---")
    clean_annotation_patterns(config.staged_final_md)
    clean_annotation_patterns(config.staged_annotated_md)

    # --- 3.7. Copy fully rendered files to base directory ---
    logger.info("--- Copying fully rendered files to base directory ---")
    shutil.copy(config.staged_final_md, config.base_final_md)
    logger.info(f"Copied final markdown to base directory: '{safe_relative_path(config.base_final_md)}'")
    shutil.copy(config.staged_annotated_md, config.base_annotated_md)
    logger.info(f"Copied annotated markdown to base directory: '{safe_relative_path(config.base_annotated_md)}'")

    # --- 4. Run Pandoc Build ---
    logger.info("--- Running Pandoc Build ---")
    metadata = {}
    if config.staged_metadata_json.exists():
        with open(config.staged_metadata_json, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            metadata["book_version"] = config.version
        with open(config.staged_metadata_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    pandoc_args = [
        "--toc",
        "--toc-depth=1",
        "--split-level=1",
        f"--css={safe_relative_path(config.epub_css)}",
    ]

    # Build Final Version
    logger.info(f"Building final version for '{config.BOOK_TITLE}'...")
    pypandoc.convert_file(
        str(config.staged_final_md),
        "epub",
        outputfile=str(config.build_final_epub),
        extra_args=pandoc_args
        + [
            f"--epub-cover-image={safe_relative_path(config.cover_image)}",
            f'--metadata=title:"{config.BOOK_TITLE}"',
            f'--metadata=author:"{config.BOOK_AUTHOR}"',
            f'--metadata=version:"{metadata["book_version"]}"',
        ],
    )
    logger.success(f"  -> Created '{safe_relative_path(config.build_final_epub)}'")

    # Build PDF from EPUB for better formatting
    logger.info("Building PDF from EPUB for final version...")
    success = build_pdf_from_epub(
        epub_path=config.build_final_epub,
        pdf_path=config.build_final_pdf,
        title=config.BOOK_TITLE,
        author=config.BOOK_AUTHOR,
        version=metadata["book_version"],
        css_path=config.epub_css,
    )
    if success:
        logger.success(f"  -> Created '{safe_relative_path(config.build_final_pdf)}'")
    else:
        logger.error(f"  -> Failed to create '{safe_relative_path(config.build_final_pdf)}'")

    # Build Annotated Version
    logger.info(f"Building annotated version for '{config.BOOK_TITLE}'...")
    pypandoc.convert_file(
        str(config.staged_annotated_md),
        "epub",
        outputfile=str(config.build_annotated_epub),
        extra_args=pandoc_args
        + [
            f"--epub-cover-image={safe_relative_path(config.cover_image)}",
            f'--metadata=title:"{config.BOOK_TITLE}: AI Edit"',
            f'--metadata=author:"{config.BOOK_AUTHOR}, LLM"',
            f'--metadata=version:"{metadata["book_version"]}"',
        ],
    )
    logger.success(f"  -> Created '{safe_relative_path(config.build_annotated_epub)}'")

    # Build PDF from EPUB for better formatting
    logger.info("Building PDF from EPUB for annotated version...")
    success = build_pdf_from_epub(
        epub_path=config.build_annotated_epub,
        pdf_path=config.build_annotated_pdf,
        title=f"{config.BOOK_TITLE}: AI Edit",
        author=f"{config.BOOK_AUTHOR}, LLM",
        version=metadata["book_version"],
        css_path=config.epub_css,
    )
    if success:
        logger.success(f"  -> Created '{safe_relative_path(config.build_annotated_pdf)}'")
    else:
        logger.error(f"  -> Failed to create '{safe_relative_path(config.build_annotated_pdf)}'")

    # --- 5. Copy Final Artifacts ---
    logger.info("--- Copying Final Artifacts to Build Directory ---")
    shutil.copy(config.staged_final_md, config.build_final_md)
    logger.info(f"Copied final markdown to '{safe_relative_path(config.build_final_md)}'")
    shutil.copy(config.staged_annotated_md, config.build_annotated_md)
    logger.info(f"Copied annotated markdown to '{safe_relative_path(config.build_annotated_md)}'")
    if config.original_file_path.exists():
        shutil.copy(config.original_file_path, config.build_original_md)
        logger.info(f"Copied original manuscript to '{safe_relative_path(config.build_original_md)}'")
    else:
        logger.warning(f"Original file not found at '{safe_relative_path(config.original_file_path)}'")

    if config.staged_metadata_json.exists():
        shutil.copy(config.staged_metadata_json, config.build_metadata_json)
        logger.info(f"Copied metadata to '{safe_relative_path(config.build_metadata_json)}'")

    # Clean up staging directory
    shutil.rmtree(config.staging_dir)
    logger.info(f"Cleaned up staging directory: '{safe_relative_path(config.staging_dir)}'")

    logger.success(f"\nBuild complete for {name} v{version}!")


def build_pdf_from_epub(epub_path: Path, pdf_path: Path, title: str, author: str, version: str, css_path: Path) -> bool:
    """
    Builds PDF from EPUB using Calibre's ebook-convert for perfect conversion.

    Args:
        epub_path: Path to the source EPUB file
        pdf_path: Path for the output PDF file
        title: Book title for metadata
        author: Book author for metadata
        version: Book version for metadata
        css_path: Path to CSS file for styling (not used with Calibre)

    Returns:
        True if PDF was successfully created, False otherwise
    """
    try:
        logger.info("Converting EPUB to PDF using Calibre ebook-convert...")

        # Use Calibre's ebook-convert for perfect EPUB to PDF conversion
        subprocess.run(["ebook-convert", str(epub_path), str(pdf_path)], capture_output=True, text=True, check=True)

        logger.success("Successfully created PDF using Calibre")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Calibre conversion failed: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(
            "Calibre ebook-convert not found. Please install Calibre and ensure ebook-convert is in your PATH."
        )
        return False
    except Exception as e:
        logger.error(f"Failed to create PDF: {str(e)}")
        return False


def main():
    """
    Main function to parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Build script for Markdown to EPUB/PDF.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the book.")
    build_parser.add_argument("version", type=str, help="The version string for the build (e.g., '1.0.0').")
    build_parser.add_argument(
        "--name",
        type=str,
        default="the_federalist_papers",
        help="The name of the book, used for directory paths.",
    )

    args = parser.parse_args()

    if args.command == "build":
        build(args.version, args.name)


if __name__ == "__main__":
    main()
