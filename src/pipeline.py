from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from src.config import PhaseType, RunConfig
from src.llm_model import LlmModel
from src.llm_phase import LlmPhase


class Pipeline:
    """Manages the execution of LLM processing phases."""

    def __init__(self, config: RunConfig):
        """Initialize the pipeline with a run configuration."""
        self.config = config
        self._phase_instances: Dict[PhaseType, LlmPhase] = {}
        self._initialize_phases()

    def _get_phase_output_path(self, phase_type: PhaseType) -> Path:
        """Generate output path for a phase."""
        phase_config = self.config.get_phase_config(phase_type)

        # Use custom output path if specified
        if phase_config.custom_output_path:
            return phase_config.custom_output_path

        # Generate default output path
        input_stem = self.config.input_file.stem

        # Remove previous phase suffixes to handle re-runs
        for pt in PhaseType:
            input_stem = input_stem.replace(f" {pt.name.capitalize()}", "")

        # Add current phase suffix
        output_stem = f"{input_stem} {phase_type.name.capitalize()}"

        return self.config.output_dir / f"{output_stem}{self.config.input_file.suffix}"

    def _get_phase_input_path(self, phase_type: PhaseType) -> Path:
        """Determine input path for a phase based on the previous phase's output."""
        phase_order = self.config.get_phase_order()
        phase_index = phase_order.index(phase_type)

        # First phase uses the run's input file
        if phase_index == 0:
            return self.config.input_file

        # Subsequent phases use the previous phase's output
        previous_phase_type = phase_order[phase_index - 1]
        return self._get_phase_output_path(previous_phase_type)

    def _initialize_phases(self) -> None:
        """Initialize all enabled phases based on the configuration."""
        for phase_type, phase_config in self.config.phases.items():
            if not phase_config.enabled:
                logger.info(f"Skipping disabled phase: {phase_type.name}")
                continue

            input_path = self._get_phase_input_path(phase_type)
            output_path = self._get_phase_output_path(phase_type)

            # Skip if input file doesn't exist
            if not input_path.exists():
                logger.warning(
                    f"Input file not found for {phase_type.name}: {input_path}"
                )
                continue

            # Initialize the model
            model = LlmModel.create(
                model_type=phase_config.model_type,
                temperature=phase_config.temperature,
            )

            # Create the phase instance
            phase = LlmPhase(
                name=phase_type.name.lower(),
                input_file_path=input_path,
                output_file_path=output_path,
                system_prompt_path=phase_config.system_prompt_path,
                book_name=self.config.book_name,
                author_name=self.config.author_name,
                model=model,
                temperature=phase_config.temperature,
                max_workers=phase_config.max_workers,
            )

            self._phase_instances[phase_type] = phase

    def run_phase(self, phase_type: PhaseType, **kwargs) -> Optional[LlmPhase]:
        """Run a specific phase."""
        if phase_type not in self._phase_instances:
            logger.warning(f"Phase not initialized or disabled: {phase_type.name}")
            return None

        phase = self._phase_instances[phase_type]
        logger.info(f"Starting phase: {phase_type.name}")

        try:
            phase.run(**kwargs)
            logger.success(f"Completed phase: {phase_type.name}")
            return phase
        except Exception as e:
            logger.error(f"Error in phase {phase_type.name}: {str(e)}")
            raise

    def run(self, **kwargs) -> None:
        """Run all enabled phases in order."""
        logger.info(
            f"Starting pipeline for {self.config.book_name} by {self.config.author_name}"
        )

        for phase_type in self.config.get_phase_order():
            if phase_type in self._phase_instances:
                self.run_phase(phase_type, **kwargs)

        logger.success("Pipeline completed successfully")
