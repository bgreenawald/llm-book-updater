import sys
from pathlib import Path

from loguru import logger

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from src.pipeline import run_pipeline
from src.run_settings import config


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
    setup_logging()
    try:
        logger.info("Starting pipeline execution from main.py")
        run_pipeline(config=config)
        logger.success("Pipeline execution finished.")
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
