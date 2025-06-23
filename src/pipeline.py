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
        # We'll initialize phases as they're needed, not all at once

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
        output_file = f"{output_stem}{self.config.input_file.suffix}"
        return self.config.output_dir / output_file

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

    def _initialize_phase(self, phase_type: PhaseType) -> Optional[LlmPhase]:
        """Initialize a single phase when it's about to run."""
        if phase_type not in self.config.phases:
            return None

        phase_config = self.config.phases[phase_type]
        if not phase_config.enabled:
            logger.info(f"Skipping disabled phase: {phase_type.name}")
            return None

        input_path = self._get_phase_input_path(phase_type)
        output_path = self._get_phase_output_path(phase_type)

        # For the first phase, check if input file exists
        phase_order = self.config.get_phase_order()
        if phase_type == phase_order[0] and not input_path.exists():
            logger.error(
                f"Input file not found for initial phase "
                f"{phase_type.name}: {input_path}"
            )
            return None

        logger.info(f"Initializing phase: {phase_type.name}")

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
        return phase

    def run_phase(self, phase_type: PhaseType, **kwargs) -> Optional[LlmPhase]:
        """Run a specific phase."""
        logger.debug(f"Attempting to run phase: {phase_type.name}")

        # Initialize the phase if not already done
        if phase_type not in self._phase_instances:
            logger.debug(f"Phase {phase_type.name} not initialized, initializing now")
            phase = self._initialize_phase(phase_type)
            if not phase:
                msg = f"Could not initialize phase: {phase_type.name}"
                logger.warning(msg)
                return None
        else:
            phase = self._phase_instances[phase_type]
            logger.debug(f"Using existing phase instance for {phase_type.name}")

        logger.info(f"Starting phase: {phase_type.name}")
        logger.debug(f"Input file: {phase.input_file_path}")
        logger.debug(f"Output file: {phase.output_file_path}")

        try:
            phase.run(**kwargs)
            logger.success(f"Successfully completed phase: {phase_type.name}")
            logger.debug(f"Output written to: {phase.output_file_path}")
            return phase
        except Exception as e:
            logger.error(f"Error in phase {phase_type.name}: {str(e)}", exc_info=True)
            raise

    def run(self, **kwargs) -> None:
        """Run all enabled phases in order."""
        phase_order = self.config.get_phase_order()
        logger.info(f"Starting pipeline with phases: {[p.name for p in phase_order]}")

        for phase_type in phase_order:
            logger.debug(f"Checking phase: {phase_type.name}")
            if phase_type not in self.config.phases:
                logger.warning(f"Phase {phase_type.name} not found in configuration")
                continue

            if not self.config.phases[phase_type].enabled:
                logger.info(f"Skipping disabled phase: {phase_type.name}")
                continue

            logger.info(f"Proceeding with phase: {phase_type.name}")
            self.run_phase(phase_type, **kwargs)

        logger.success("Pipeline completed successfully")
