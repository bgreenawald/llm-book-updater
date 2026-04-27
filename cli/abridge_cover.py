"""
Abridged cover generation command module.

This module handles generating a cover for the abridged edition of a book,
based on the book's existing cover image.
"""

import sys
from pathlib import Path

import click

from .common import convert_to_webp, get_available_books, validate_book_exists
from .cover import generate_cover_image, get_book_metadata

ABRIDGE_COVER_PROMPT = """\
You are creating an **abridged edition variant** of the book cover shown in \
the reference image.

The reference image is the original cover for this book. Your task is to \
produce a new cover that:

1. **Mirrors the original design closely** — same color scheme, same \
three-block horizontal layout, same overall proportions and visual style.
2. **Updates the bottom block** — replace any edition label (e.g. \
"Timeless Library Edition") with **"Abridged Edition"**.
3. **Preserves the title and author** — keep them exactly as they appear, \
using the same typography style and placement.
4. **Keeps the visual element** — retain the same illustrative or geometric \
element from the top block; do not change it.

Title: {title}
Author: {author}

The result must be clearly recognizable as the abridged companion to the \
original cover while being visually consistent with the series aesthetic.
"""


@click.command("abridge-cover")
@click.argument("book_name", required=True)
@click.option(
    "--model", type=str, default="google/gemini-3-pro-image-preview", help="OpenRouter model for image generation"
)
@click.help_option("--help", "-h")
def abridge_cover_command(book_name: str, model: str) -> None:
    """
    Generate an abridged edition cover based on the book's existing cover.

    BOOK_NAME: Name of the book to generate an abridged cover for (supports partial matching)

    The command reads the book's existing cover.png as a reference and produces
    a variant cover labelled "Abridged Edition", saving it as abridge-cover.png
    and abridge-cover.webp in the book directory.

    Examples:
      python -m cli abridge-cover on_liberty
      python -m cli abridge-cover liberty --model google/gemini-2.5-flash-image-preview
    """
    available_books = get_available_books()

    if not available_books:
        click.echo("No books found")
        sys.exit(1)

    matched_book_name = validate_book_exists(book_name, available_books)

    click.echo(f"Generating abridged cover for '{matched_book_name}'...")

    # Ensure the source cover exists
    source_cover_path = Path("books") / matched_book_name / "cover.png"
    if not source_cover_path.exists():
        click.echo(f"Error: No existing cover found at {source_cover_path}", err=True)
        click.echo("Run 'python -m cli cover' first to generate the original cover.", err=True)
        sys.exit(1)

    try:
        title, author = get_book_metadata(matched_book_name)
        click.echo(f"Title: {title}")
        click.echo(f"Author: {author}")

        prompt = ABRIDGE_COVER_PROMPT.format(title=title, author=author)

        from llm_core.config import settings

        api_key = settings.get_api_key("openrouter")

        if not api_key:
            click.echo("Error: OPENROUTER_API_KEY not found in environment", err=True)
            sys.exit(1)

        image_data = generate_cover_image(prompt, source_cover_path, api_key, model)

        # Save PNG
        png_output_path = Path("books") / matched_book_name / "abridge-cover.png"
        png_output_path.write_bytes(image_data)

        # Save WebP
        click.echo("Converting cover to WebP format...")
        webp_image = convert_to_webp(image_data)
        webp_output_path = Path("books") / matched_book_name / "abridge-cover.webp"
        webp_output_path.write_bytes(webp_image)

        click.echo(f"Abridged cover saved to: {png_output_path}")
        click.echo(f"Abridged cover saved to: {webp_output_path}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)
