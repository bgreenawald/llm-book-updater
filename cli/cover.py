"""
Cover generation command module.

This module handles generating book covers using OpenRouter's image generation API.
"""

import base64
import sys
from pathlib import Path

import click
import requests

from .common import convert_to_webp, get_available_books, validate_book_exists


def get_book_metadata(book_name: str) -> tuple[str, str]:
    """
    Extract title and author from book's build.py by parsing the file.

    Args:
        book_name: Name of the book directory

    Returns:
        Tuple of (title, author)

    Raises:
        FileNotFoundError: If build.py doesn't exist
        ValueError: If title/author cannot be extracted
    """
    build_path = Path("books") / book_name / "build.py"

    if not build_path.exists():
        raise FileNotFoundError(f"build.py not found for book '{book_name}'")

    content = build_path.read_text()

    # Parse the file to extract title and author from BookConfig
    title = None
    author = None

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("title="):
            # Extract the title string
            title = line.split("title=")[1].strip().strip(",").strip('"').strip("'")
        elif line.startswith("author="):
            # Extract the author string
            author = line.split("author=")[1].strip().strip(",").strip('"').strip("'")

    if not title or not author:
        raise ValueError(f"Could not extract title and author from {build_path}")

    return title, author


def load_prompt_template() -> str:
    """
    Load the book cover generation prompt template.

    Returns:
        The prompt template text

    Raises:
        FileNotFoundError: If the prompt template doesn't exist
    """
    prompt_path = Path("assets/book_cover_prompt.md")

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found at {prompt_path}")

    return prompt_path.read_text()


def encode_image_to_base64(image_path: Path) -> str:
    """
    Encode an image file to base64.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_cover_image(
    prompt: str, reference_image_path: Path, api_key: str, model: str = "google/gemini-3-pro-image-preview"
) -> bytes:
    """
    Generate a cover image using OpenRouter's image generation API.

    Args:
        prompt: The generation prompt
        reference_image_path: Path to the reference cover image
        api_key: OpenRouter API key
        model: Model to use for generation

    Returns:
        Image data as bytes

    Raises:
        requests.exceptions.RequestException: On API errors
        ValueError: If the response doesn't contain an image
    """
    # Encode the reference image
    base64_image = encode_image_to_base64(reference_image_path)

    # Determine the image MIME type
    suffix = reference_image_path.suffix.lower()
    mime_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
        suffix, "image/png"
    )

    # Build the API request
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
            "aspect_ratio": "2:3"  # Book cover ratio
        },
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    click.echo("Generating cover image...")

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


@click.command("cover")
@click.argument("book_name", required=True)
@click.option(
    "--example-cover",
    type=click.Path(exists=True),
    default="books/on_liberty/cover.png",
    help="Path to example cover for style reference",
)
@click.option(
    "--model", type=str, default="google/gemini-3-pro-image-preview", help="OpenRouter model for image generation"
)
@click.help_option("--help", "-h")
def cover_command(book_name: str, example_cover: str, model: str) -> None:
    """
    Generate a book cover using AI image generation.

    BOOK_NAME: Name of the book to generate a cover for (supports partial matching)

    Examples:
      python -m cli cover on_liberty
      python -m cli cover liberty --model google/gemini-2.5-flash-image-preview
      python -m cli cover the_federalist --example-cover books/democracy_in_america/cover.png
    """
    # Validate book exists
    available_books = get_available_books()

    if not available_books:
        click.echo("No books found")
        sys.exit(1)

    matched_book_name = validate_book_exists(book_name, available_books)

    click.echo(f"Generating cover for '{matched_book_name}'...")

    try:
        # Get book metadata
        title, author = get_book_metadata(matched_book_name)
        click.echo(f"Title: {title}")
        click.echo(f"Author: {author}")

        # Load prompt template
        prompt_template = load_prompt_template()

        # Build the full prompt with book-specific details
        full_prompt = f"""Generate a book cover for:

Title: {title}
Author: {author}

{prompt_template}"""

        # Get API key from environment
        from llm_core.config import settings

        api_key = settings.get_api_key("openrouter")

        if not api_key:
            click.echo("Error: OPENROUTER_API_KEY not found in environment", err=True)
            sys.exit(1)

        # Generate the cover
        reference_path = Path(example_cover)
        image_data = generate_cover_image(full_prompt, reference_path, api_key, model)

        # Save PNG
        png_output_path = Path("books") / matched_book_name / "cover.png"
        png_output_path.write_bytes(image_data)

        # Save WebP
        click.echo("Converting cover to WebP format...")
        webp_image = convert_to_webp(image_data)
        webp_output_path = Path("books") / matched_book_name / "cover.webp"
        webp_output_path.write_bytes(webp_image)

        click.echo(f"Cover saved to: {png_output_path}")
        click.echo(f"Cover saved to: {webp_output_path}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)
