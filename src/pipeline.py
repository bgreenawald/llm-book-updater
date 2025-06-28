import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger

from src.config import PhaseType, RunConfig
from src.llm_model import LlmModel, LlmModelError
from src.llm_phase import IntroductionAnnotationPhase, LlmPhase, StandardLlmPhase, SummaryAnnotationPhase
from src.phase_factory import PhaseFactory


def create_phase(phase_type: PhaseType, **kwargs) -> LlmPhase:
    """
    Factory function to create the appropriate phase instance based on PhaseType.

    Args:
        phase_type: The type of phase to create
        **kwargs: Arguments to pass to the phase constructor

    Returns:
        An instance of the appropriate phase class

    Raises:
        ValueError: If the phase type is not supported
    """
    phase_mapping = {
        PhaseType.MODERNIZE: StandardLlmPhase,
        PhaseType.EDIT: StandardLlmPhase,
        PhaseType.ANNOTATE: StandardLlmPhase,
        PhaseType.FINAL: StandardLlmPhase,
        PhaseType.INTRODUCTION: IntroductionAnnotationPhase,
        PhaseType.SUMMARY: SummaryAnnotationPhase,
    }

    phase_class = phase_mapping.get(phase_type)
    if phase_class is None:
        raise ValueError(f"Unsupported phase type: {phase_type}")

    return phase_class(**kwargs)


class Pipeline:
    """Manages the execution of LLM processing phases."""

    def __init__(self, config: RunConfig):
        """Initialize the pipeline with a run configuration."""
        self.config = config
        self._phase_instances: List[LlmPhase] = []

    def __str__(self):
        return f"Pipeline(config={self.config})"

    def __repr__(self):
        return f"Pipeline(config={self.config})"

    def _save_run_metadata(self, completed_phases: List[LlmPhase]) -> None:
        """Save metadata about the pipeline run to the output directory."""
        metadata = {
            "run_timestamp": datetime.now().isoformat(),
            "book_name": self.config.book_name,
            "author_name": self.config.author_name,
            "input_file": str(self.config.input_file),
            "original_file": str(self.config.original_file),
            "output_directory": str(self.config.output_dir),
            "phases": [],
        }

        # Add information about each completed phase
        for i, phase in enumerate(completed_phases):
            phase_config = self.config.phases[i]

            # Get post-processor information
            post_processors_info = []
            if phase.post_processor_chain:
                post_processors_info = [p.name for p in phase.post_processor_chain.processors]

            phase_metadata = {
                "phase_name": phase.name,
                "phase_index": i,
                "enabled": phase_config.enabled,
                "model_type": phase_config.model_type,
                "temperature": phase_config.temperature,
                "input_file": str(phase.input_file_path),
                "output_file": str(phase.output_file_path),
                "system_prompt": str(phase.system_prompt_path) if phase.system_prompt_path else None,
                "user_prompt": str(phase.user_prompt_path) if phase.user_prompt_path else None,
                "max_workers": phase_config.max_workers,
                "post_processors": post_processors_info,
                "post_processor_count": len(post_processors_info),
                "completed": True,
                "output_exists": phase.output_file_path.exists() if phase.output_file_path else False,
            }
            metadata["phases"].append(phase_metadata)

        # Add information about phases that were configured but not run
        for i, phase_config in enumerate(self.config.phases):
            if i >= len(completed_phases):
                # Get post-processor information from config
                post_processors_info = []
                if phase_config.post_processors:
                    # Convert post-processors to readable names
                    for processor in phase_config.post_processors:
                        if hasattr(processor, "name"):
                            post_processors_info.append(processor.name)
                        elif isinstance(processor, str):
                            post_processors_info.append(processor)
                        else:
                            post_processors_info.append(str(processor))

                phase_metadata = {
                    "phase_name": phase_config.phase_type.name.lower(),
                    "phase_index": i,
                    "enabled": phase_config.enabled,
                    "model_type": phase_config.model_type,
                    "temperature": phase_config.temperature,
                    "max_workers": phase_config.max_workers,
                    "post_processors": post_processors_info,
                    "post_processor_count": len(post_processors_info),
                    "completed": False,
                    "reason": "disabled" if not phase_config.enabled else "not_run",
                }
                metadata["phases"].append(phase_metadata)

        # Save metadata to output directory
        metadata_file = self.config.output_dir / f"run_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Run metadata saved to: {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save run metadata: {str(e)}")

    def _get_phase_output_path(self, phase_index: int) -> Path:
        """Generate output path for a phase."""
        phase_config = self.config.phases[phase_index]

        # Use custom output path if specified
        if phase_config.custom_output_path:
            return phase_config.custom_output_path

        # Generate default output path
        input_stem = self.config.input_file.stem
        phase_type = phase_config.phase_type

        # Count occurrences of this phase type up to the current index
        phase_count = sum(1 for i in range(phase_index + 1) if self.config.phases[i].phase_type == phase_type)

        # Add phase index (1-based) and current phase suffix with count
        phase_index_1_based = phase_index + 1
        output_stem = f"{phase_index_1_based:02d}-{input_stem} {phase_type.name.capitalize()}_{phase_count}"
        output_file = f"{output_stem}{self.config.input_file.suffix}"
        return self.config.output_dir / output_file

    def _get_phase_input_path(self, phase_index: int) -> Path:
        """Determine input path for a phase based on the previous phase's output."""
        # First phase uses the run's input file
        if phase_index == 0:
            return self.config.input_file

        # Subsequent phases use the previous phase's output
        previous_phase_index = phase_index - 1
        return self._get_phase_output_path(previous_phase_index)

    def _initialize_phase(self, phase_index: int) -> Optional[LlmPhase]:
        """Initialize a single phase when it's about to run."""
        phase_config = self.config.phases[phase_index]
        if not phase_config.enabled:
            logger.info(f"Skipping disabled phase: {phase_config.phase_type.name}")
            return None

        input_path = self._get_phase_input_path(phase_index)
        output_path = self._get_phase_output_path(phase_index)

        # For the first phase, check if input file exists
        if phase_index == 0 and not input_path.exists():
            logger.error(f"Input file not found for initial phase {phase_config.phase_type.name}: {input_path}")
            return None

        logger.info(f"Initializing phase: {phase_config.phase_type.name} (run {phase_index + 1})")

        # Initialize the model
        model = LlmModel.create(
            model=phase_config.model_type,
            temperature=phase_config.temperature,
        )

        # Create PhaseConfig for the factory
        factory_config = type(phase_config)(
            phase_type=phase_config.phase_type,
            name=phase_config.phase_type.name.lower(),
            input_file_path=input_path,
            output_file_path=output_path,
            original_file_path=self.config.original_file,
            system_prompt_path=phase_config.system_prompt_path,
            user_prompt_path=phase_config.user_prompt_path,
            book_name=self.config.book_name,
            author_name=self.config.author_name,
            model=model,
            temperature=phase_config.temperature,
            max_workers=phase_config.max_workers,
            reasoning=phase_config.reasoning,
            post_processors=phase_config.post_processors,
        )

        # Create the phase instance using the factory
        if phase_config.phase_type in [PhaseType.MODERNIZE, PhaseType.EDIT, PhaseType.FINAL, PhaseType.ANNOTATE]:
            phase = PhaseFactory.create_standard_phase(factory_config)
        elif phase_config.phase_type == PhaseType.INTRODUCTION:
            phase = PhaseFactory.create_introduction_annotation_phase(factory_config)
        elif phase_config.phase_type == PhaseType.SUMMARY:
            phase = PhaseFactory.create_summary_annotation_phase(factory_config)
        else:
            raise ValueError(f"Unsupported phase type: {phase_config.phase_type}")

        # Log post-processor information
        if phase.post_processor_chain:
            processor_names = [p.name for p in phase.post_processor_chain.processors]
            logger.info(f"Post-processing pipeline for {phase.name}: {processor_names}")
            logger.info(f"Post-processor count: {len(processor_names)}")
        else:
            logger.info(f"No post-processors configured for {phase.name}")

        return phase

    def run(self, **kwargs) -> None:
        """Run all enabled phases in order."""
        phase_order = self.config.get_phase_order()
        logger.info(f"Starting pipeline with phases: {[p.name for p in phase_order]}")

        completed_phases: List[LlmPhase] = []

        try:
            for i, phase_config in enumerate(self.config.phases):
                if not phase_config.enabled:
                    logger.info(f"Skipping disabled phase: {phase_config.phase_type.name}")
                    continue

                logger.info(f"Proceeding with phase: {phase_config.phase_type.name}")

                phase = self._initialize_phase(i)
                if not phase:
                    logger.warning(f"Could not initialize phase: {phase_config.phase_type.name}")
                    continue

                self._phase_instances.append(phase)

                try:
                    logger.info(f"Starting phase: {phase.name}")
                    logger.debug(f"Input file: {phase.input_file_path}")
                    logger.debug(f"Output file: {phase.output_file_path}")
                    phase.run(**kwargs)
                    logger.success(f"Successfully completed phase: {phase.name}")
                    logger.debug(f"Output written to: {phase.output_file_path}")
                    completed_phases.append(phase)
                except LlmModelError as e:
                    logger.error(f"LLM model error in phase {phase.name}: {str(e)}")
                    logger.error("Pipeline stopped due to LLM model failure after max retries")
                    raise
                except Exception as e:
                    logger.error(f"Error in phase {phase.name}: {str(e)}", exc_info=True)
                    raise

            logger.success("Pipeline completed successfully")
        finally:
            # Save metadata about the run (whether successful or failed)
            self._save_run_metadata(completed_phases)


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
