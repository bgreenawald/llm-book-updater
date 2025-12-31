"""Pandoc conversion utilities for PDF and EPUB generation."""

import shutil
import subprocess
from pathlib import Path


class ConversionError(Exception):
    """Error during format conversion."""

    pass


def check_pandoc_installed() -> bool:
    """Check if Pandoc is installed and available."""
    return shutil.which("pandoc") is not None


def convert_to_pdf(input_md: Path, output_pdf: Path) -> Path:
    """
    Convert markdown to PDF using Pandoc.

    Requires Pandoc and a LaTeX distribution (e.g., texlive, xelatex).
    """
    if not check_pandoc_installed():
        raise ConversionError("Pandoc not found. Install Pandoc from https://pandoc.org/installing.html")

    # Check for LaTeX
    if not shutil.which("xelatex") and not shutil.which("pdflatex"):
        raise ConversionError("LaTeX not found. Install a LaTeX distribution (e.g., texlive-xetex) for PDF generation.")

    cmd = [
        "pandoc",
        str(input_md),
        "-o",
        str(output_pdf),
        "--pdf-engine=xelatex",
        "--toc",
        "--toc-depth=2",
        "-V",
        "geometry:margin=1in",
        "-V",
        "documentclass=book",
        "-V",
        "fontsize=11pt",
        "-V",
        "linkcolor=blue",
        "-V",
        "urlcolor=blue",
        "--highlight-style=tango",
    ]

    # Try xelatex first, fall back to pdflatex
    if not shutil.which("xelatex"):
        cmd[cmd.index("--pdf-engine=xelatex")] = "--pdf-engine=pdflatex"

    try:
        _ = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return output_pdf
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or "Unknown error"
        raise ConversionError(f"PDF conversion failed: {error_msg}") from e


def convert_to_epub(input_md: Path, output_epub: Path) -> Path:
    """Convert markdown to EPUB using Pandoc."""
    if not check_pandoc_installed():
        raise ConversionError("Pandoc not found. Install Pandoc from https://pandoc.org/installing.html")

    cmd = [
        "pandoc",
        str(input_md),
        "-o",
        str(output_epub),
        "--toc",
        "--toc-depth=2",
        "--epub-chapter-level=1",
        "--highlight-style=tango",
    ]

    try:
        _ = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return output_epub
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or "Unknown error"
        raise ConversionError(f"EPUB conversion failed: {error_msg}") from e


def convert_to_html(input_md: Path, output_html: Path) -> Path:
    """Convert markdown to standalone HTML using Pandoc."""
    if not check_pandoc_installed():
        raise ConversionError("Pandoc not found. Install Pandoc from https://pandoc.org/installing.html")

    cmd = [
        "pandoc",
        str(input_md),
        "-o",
        str(output_html),
        "--standalone",
        "--toc",
        "--toc-depth=2",
        "--highlight-style=tango",
        "-c",
        "https://cdn.jsdelivr.net/npm/water.css@2/out/water.css",
    ]

    try:
        _ = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return output_html
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or "Unknown error"
        raise ConversionError(f"HTML conversion failed: {error_msg}") from e


def get_pandoc_version() -> str | None:
    """Get the installed Pandoc version."""
    if not check_pandoc_installed():
        return None

    try:
        result = subprocess.run(
            ["pandoc", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        # First line is "pandoc X.Y.Z"
        first_line = result.stdout.split("\n")[0]
        return first_line.replace("pandoc ", "")
    except subprocess.CalledProcessError:
        return None
