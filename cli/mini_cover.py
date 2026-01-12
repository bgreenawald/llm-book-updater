"""
Mini cover generation command module.

This module handles creating thumbnail versions of book covers by cropping the
top block and using AI to remove text and extend borders.
"""

import base64
import io
import sys
from pathlib import Path

import click
import requests
from PIL import Image

from .common import get_available_books, validate_book_exists


def encode_image_to_base64(image_data: bytes) -> str:
    """
    Encode image bytes to base64.

    Args:
        image_data: Image data as bytes

    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_data).decode("utf-8")


def process_mini_cover(cover_path: Path, api_key: str, model: str = "google/gemini-3-pro-image-preview") -> bytes:
    """
    Send full cover to AI to identify the top block, crop it, remove text, and extend borders.

    Args:
        cover_path: Path to the full cover image
        api_key: OpenRouter API key
        model: Model to use for generation

    Returns:
        Processed image bytes

    Raises:
        requests.exceptions.RequestException: On API errors
        ValueError: If the response doesn't contain an image
    """
    # Load and encode the full cover image
    with open(cover_path, "rb") as f:
        cover_data = f.read()

    base64_image = encode_image_to_base64(cover_data)

    # Determine the image MIME type
    suffix = cover_path.suffix.lower()
    mime_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
        suffix, "image/png"
    )

    prompt = """This is a book cover with a three-block design.
    The top block (approximately 60% of the image) contains the book title.

Please:
1. Identify and extract only the top title block (approximately the top 60% of the cover)
2. Remove the text from this block while preserving all other design elements
3. If a border exists, extend the border to all sides.
4. Return a square thumbnail suitable for use as a mini cover
5. ONLY REMOVE TEXT. Do not remove other visual elements.

The result should be a clean, text-free version of the original images top square."""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                ],
            }
        ],
        "modalities": ["image", "text"],
        "image_config": {
            "aspect_ratio": "1:1"  # Square thumbnail
        },
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    click.echo("Processing mini cover (identifying top block, removing text, and extending borders)...")

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120
        )
        response.raise_for_status()

        result = response.json()

        # Extract the generated image from the response
        if not result.get("choices"):
            raise ValueError("No choices in API response")

        message = result["choices"][0].get("message", {})
        images = message.get("images", [])

        if not images:
            raise ValueError("No images in API response")

        # Get the first image
        image_data = images[0].get("image_url", {}).get("url", "")

        if not image_data.startswith("data:"):
            raise ValueError("Invalid image data format")

        # Decode the base64 image data
        # Format: data:image/png;base64,<base64_data>
        base64_data = image_data.split(",", 1)[1]
        return base64.b64decode(base64_data)

    except requests.exceptions.RequestException as e:
        click.echo(f"API request failed: {e}", err=True)
        raise
    except (KeyError, ValueError, IndexError) as e:
        click.echo(f"Failed to parse API response: {e}", err=True)
        raise


def convert_to_webp(image_data: bytes) -> bytes:
    """
    Convert image bytes to WebP format.

    Args:
        image_data: Image data as bytes

    Returns:
        WebP image data as bytes
    """
    with Image.open(io.BytesIO(image_data)) as img:
        buffer = io.BytesIO()
        img.save(buffer, format="WEBP", quality=90)
        return buffer.getvalue()


@click.command("mini-cover")
@click.argument("book_name", required=True)
@click.option(
    "--model", type=str, default="google/gemini-3-pro-image-preview", help="OpenRouter model for image generation"
)
@click.help_option("--help", "-h")
def mini_cover_command(book_name: str, model: str) -> None:
    """
    Generate a mini cover (thumbnail) from an existing book cover.

    BOOK_NAME: Name of the book to generate a mini cover for (supports partial matching)

    This command takes the existing cover.png and uses AI to identify the top block,
    crop it, remove the text, extend the borders, and save as mini-cover.webp.

    Examples:
      python -m cli mini-cover on_liberty
      python -m cli mini-cover the_road_to_serfdom
      python -m cli mini-cover serfdom --model google/gemini-2.5-flash-image-preview
    """
    # Validate book exists
    available_books = get_available_books()

    if not available_books:
        click.echo("No books found")
        sys.exit(1)

    matched_book_name = validate_book_exists(book_name, available_books)

    click.echo(f"Generating mini cover for '{matched_book_name}'...")

    # Check if cover.png exists
    cover_path = Path("books") / matched_book_name / "cover.png"

    if not cover_path.exists():
        click.echo(f"Error: cover.png not found for book '{matched_book_name}'", err=True)
        click.echo("Please run 'python -m cli cover' first to generate the main cover", err=True)
        sys.exit(1)

    try:
        # Step 1: Get API key
        from llm_core.config import settings

        api_key = settings.get_api_key("openrouter")

        if not api_key:
            click.echo("Error: OPENROUTER_API_KEY not found in environment", err=True)
            sys.exit(1)

        # Step 2: Process with AI to identify top block, crop, remove text, and extend borders
        processed_image = process_mini_cover(cover_path, api_key, model)

        # Step 3: Convert to WebP
        click.echo("Converting to WebP format...")
        webp_image = convert_to_webp(processed_image)

        # Step 4: Save the mini cover
        output_path = Path("books") / matched_book_name / "mini-cover.webp"
        output_path.write_bytes(webp_image)

        click.echo(f"Mini cover saved to: {output_path}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)
