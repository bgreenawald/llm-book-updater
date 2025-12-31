"""Configuration management for the book writer application."""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic_settings import BaseSettings

from .models import BookConfig, GenerationConfig


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    openrouter_api_key: str = ""
    default_model: str = "anthropic/claude-sonnet-4"
    max_retries: int = 3
    max_concurrent_chapters: int = 5

    class Config:
        env_file = ".env"
        env_prefix = ""
        extra = "ignore"


def get_settings() -> Settings:
    """Load and validate settings from environment."""
    return Settings()


def load_book_config(book_dir: Path) -> BookConfig:
    """Load book-specific configuration from config.yaml if it exists."""
    config_file = book_dir / "config.yaml"

    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data:
                return BookConfig.model_validate(data)

    # Return default config if no file exists
    return BookConfig()


def save_book_config(book_dir: Path, config: BookConfig) -> None:
    """Save book configuration to config.yaml."""
    config_file = book_dir / "config.yaml"

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)


def get_generation_config(
    book_dir: Path,
    model_override: Optional[str] = None,
    max_concurrent_override: Optional[int] = None,
) -> GenerationConfig:
    """
    Build generation config with proper priority:
    1. CLI overrides (highest)
    2. Book config.yaml
    3. Environment variables
    4. Defaults (lowest)
    """
    # Load from environment
    settings = get_settings()

    # Load book-specific config
    book_config = load_book_config(book_dir)

    # Build config with priority chain
    return GenerationConfig(
        model=model_override or book_config.model or settings.default_model,
        max_retries=settings.max_retries,
        base_delay=1.0,
        max_delay=60.0,
        max_concurrent_chapters=(
            max_concurrent_override or book_config.max_concurrent_chapters or settings.max_concurrent_chapters
        ),
    )


def get_api_key() -> str:
    """Get the OpenRouter API key from environment."""
    settings = get_settings()
    if not settings.openrouter_api_key:
        # Try direct environment variable as fallback
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            return api_key
        raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
    return settings.openrouter_api_key


def validate_book_directory(book_dir: Path) -> Path:
    """Validate that a book directory exists and has required files."""
    if not book_dir.exists():
        raise ValueError(f"Book directory does not exist: {book_dir}")

    rubric_file = book_dir / "rubric.md"
    if not rubric_file.exists():
        raise ValueError(f"Rubric file not found: {rubric_file}")

    return book_dir


def ensure_output_directory(book_dir: Path) -> Path:
    """Ensure the output directory exists."""
    output_dir = book_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    chapters_dir = output_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    return output_dir
