import sys
from pathlib import Path

from loguru import logger


def setup_logging() -> None:
    """Configure logging with debug level and better formatting."""
    logger.remove()  # Remove default handler
    logger.add(
        sink=sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )


def main() -> None:
    """Main function to run the pipeline."""
    from book_updater.pipeline import run_pipeline

    setup_logging()
    try:
        logger.info("Starting pipeline execution from main.py")
        # Note: config must be defined here or imported from a module
        # This is a legacy entry point - consider using 'python -m cli run <book_name>' instead
        # Example: Define a RunConfig here or import from your book's run.py module
        logger.error(
            "main.py requires a config object to be defined. "
            "Please use 'python -m cli run <book_name>' or define a RunConfig in this file."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
