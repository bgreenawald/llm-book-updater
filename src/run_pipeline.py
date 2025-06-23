from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from src.config import PhaseType, RunConfig
from src.pipeline import Pipeline


def create_default_run(
    book_name: str,
    author_name: str,
    input_file: str,
    original_file: str,
    output_dir: str,
    custom_phases: Optional[Dict[PhaseType, Dict]] = None,
) -> RunConfig:
    """
    Create a default run configuration with optional overrides.

    Args:
        book_name: Name of the book
        author_name: Name of the author
        input_file: Path to the input file
        original_file: Path to the original file
        output_dir: Directory to store output files
        custom_phases: Dictionary of phase overrides
            Example: {
                PhaseType.MODERNIZE: {
                    'enabled': True,
                    'model_type': ModelType.GEMINI_FLASH,
                    'temperature': 0.2,
                    'custom_output_path': 'path/to/custom_output.md',
                },
                ...
            }
    """
    # Convert string paths to Path objects
    input_path = Path(input_file)
    output_path = Path(output_dir)

    # Create base configuration
    config = RunConfig(
        book_name=book_name,
        author_name=author_name,
        input_file=input_path,
        output_dir=output_path,
    )

    # Apply custom phase configurations if provided
    if custom_phases:
        for phase_type, phase_overrides in custom_phases.items():
            if phase_type in config.phases:
                for key, value in phase_overrides.items():
                    if hasattr(config.phases[phase_type], key):
                        setattr(config.phases[phase_type], key, value)

    return config


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
    # Example usage
    config = create_default_run(
        book_name="On Liberty",
        author_name="John Stuart Mill",
        input_file=r"books\On Liberty\markdown\Mill, On Liberty\Mill, On Liberty Clean.md",
        output_dir=r"books\On Liberty\markdown\Mill, On Liberty",
        custom_phases={
            # Example: Disable the edit phase
            # PhaseType.EDIT: {'enabled': False},
            # Example: Use a different model for annotation
            # PhaseType.ANNOTATE: {'model_type': ModelType.GEMINI_PRO},
        },
    )

    run_pipeline(config)
