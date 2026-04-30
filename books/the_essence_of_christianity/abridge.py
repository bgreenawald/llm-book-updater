import sys

from book_updater import create_abridge_run_config
from book_updater.logging_config import setup_logging
from book_updater.pipeline import run_pipeline

config = create_abridge_run_config(
    book_id="the_essence_of_christianity",
    book_name="The Essence of Christianity",
    author_name="Ludwig Feuerbach",
    max_workers=3,
)


def main() -> None:
    """Main function to run the abridged pipeline."""
    logger = setup_logging("the_essence_of_christianity")
    try:
        logger.info("Starting abridged pipeline execution")
        run_pipeline(config=config)
        logger.success("Abridged pipeline execution finished.")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"An error occurred during pipeline execution: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
