from pathlib import Path
from typing import List

from loguru import logger

from src.config import PhaseConfig, PhaseType, RunConfig
from src.llm_model import OPENAI_04_MINI
from src.pipeline import Pipeline


def run_pipeline(config: RunConfig) -> None:
    """Run the pipeline with the given configuration."""
    logger.info(f"Running pipeline for {config.book_name}")

    try:
        pipeline = Pipeline(config)
        pipeline.run()
        logger.success("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    from src.run_settings import config

    run_pipeline(config)