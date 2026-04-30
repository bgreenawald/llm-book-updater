import sys

from book_updater import create_abridge_run_config
from book_updater.logging_config import setup_logging
from book_updater.pipeline import run_pipeline

config = create_abridge_run_config(
    book_id="the_use_of_knowledge_in_society",
    book_name="The Use of Knowledge in Society",
    author_name="Friedrich Hayek",
)


def main() -> None:
    """Main function to run the abridged pipeline."""
    logger = setup_logging("the_use_of_knowledge_in_society")
    try:
        logger.info("Starting abridged pipeline execution")
        run_pipeline(config=config)
        logger.success("Abridged pipeline execution finished.")
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
