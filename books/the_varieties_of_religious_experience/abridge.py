import sys

from book_updater import create_abridge_run_config
from book_updater.logging_config import setup_logging
from book_updater.pipeline import run_pipeline

config = create_abridge_run_config(
    book_id="the_varieties_of_religious_experience",
    book_name="The Varieties of Religious Experience",
    author_name="William James",
    max_workers=5,
)


def main() -> None:
    """Main function to run the abridged pipeline."""
    logger = setup_logging("the_varieties_of_religious_experience")
    try:
        logger.info("Starting abridged pipeline execution")
        run_pipeline(config=config)
        logger.success("Abridged pipeline execution finished.")
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
