import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.config import PhaseType, RunConfig
from src.llm_model import LlmModel, LlmModelError
from src.llm_phase import IntroductionAnnotationPhase, LlmPhase, StandardLlmPhase, SummaryAnnotationPhase
from src.logging_config import setup_logging
from src.phase_factory import PhaseFactory

# Initialize module-level logger
module_logger = setup_logging(log_name="pipeline")

# Metadata version for compatibility
METADATA_VERSION = "1.0.0"


def create_phase(phase_type: PhaseType, **kwargs) -> LlmPhase:
    """
    Factory function to create the appropriate phase instance based on PhaseType.

    Args:
        phase_type (PhaseType): The type of phase to create
        **kwargs: Arguments to pass to the phase constructor

    Returns:
        LlmPhase: An instance of the appropriate phase class

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
    """
    Manages the execution of LLM processing phases.

    The Pipeline class orchestrates the sequential execution of multiple
    LLM processing phases, handling file I/O, phase initialization,
    and metadata collection throughout the process.
    """

    def __init__(self, config: RunConfig):
        """
        Initialize the pipeline with a run configuration.

        Args:
            config (RunConfig): Configuration object containing all run parameters
        """
        self.config = config
        self._phase_instances: List[LlmPhase] = []
        self._phase_metadata: List[Dict[str, Any]] = []  # Collect comprehensive metadata for all phases

    def __str__(self) -> str:
        """
        Returns a string representation of the Pipeline instance.

        Returns:
            str: String representation of the pipeline
        """
        return f"Pipeline(config={self.config})"

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the Pipeline instance for debugging.

        Returns:
            str: Detailed string representation of the pipeline
        """
        return f"Pipeline(config={self.config})"

    def _collect_phase_metadata(self, phase: Optional[LlmPhase], phase_index: int, completed: bool = False) -> None:
        """
        Collect comprehensive metadata about a phase.

        This method collects detailed metadata about a phase including configuration,
        model settings, prompt information, and execution status. The metadata is
        stored for later saving.

        Args:
            phase (Optional[LlmPhase]): The phase instance to collect metadata from,
                or None for disabled/failed phases
            phase_index (int): The index of the phase in the pipeline sequence
            completed (bool): Whether the phase completed successfully
        """
        phase_config = self.config.phases[phase_index]

        # Get post-processor information
        post_processors_info = []
        if phase and phase.post_processor_chain:
            post_processors_info = [p.name for p in phase.post_processor_chain.processors]
        elif phase_config.post_processors:
            # Convert post-processors to readable names
            for processor in phase_config.post_processors:
                if hasattr(processor, "name"):
                    post_processors_info.append(processor.name)
                elif isinstance(processor, str):
                    post_processors_info.append(processor)
                else:
                    post_processors_info.append(str(processor))

        # Base metadata that's always available
        metadata = {
            "phase_name": phase.name if phase else phase_config.phase_type.name.lower(),
            "phase_index": phase_index,
            "phase_type": phase_config.phase_type.name,
            "enabled": phase_config.enabled,
            "model_type": phase_config.model_type,
            "temperature": phase_config.temperature,
            "max_workers": phase_config.max_workers,
            "post_processors": post_processors_info,
            "post_processor_count": len(post_processors_info),
            "completed": completed,
            "book_name": self.config.book_name,
            "author_name": self.config.author_name,
        }

        # Add phase-specific information if phase exists
        if phase:
            metadata.update(
                {
                    "input_file": str(phase.input_file_path),
                    "output_file": str(phase.output_file_path),
                    "system_prompt_path": str(phase.system_prompt_path) if phase.system_prompt_path else None,
                    "user_prompt_path": str(phase.user_prompt_path) if phase.user_prompt_path else None,
                    "fully_rendered_system_prompt": phase.system_prompt,
                    "length_reduction_parameter": phase.length_reduction,
                    "output_exists": phase.output_file_path.exists() if phase.output_file_path else False,
                }
            )
        else:
            # For disabled/failed phases, add default values
            metadata.update(
                {
                    "input_file": None,
                    "output_file": None,
                    "system_prompt_path": str(phase_config.system_prompt_path)
                    if phase_config.system_prompt_path
                    else None,
                    "user_prompt_path": str(phase_config.user_prompt_path) if phase_config.user_prompt_path else None,
                    "fully_rendered_system_prompt": None,
                    "length_reduction_parameter": self.config.length_reduction,
                    "output_exists": False,
                }
            )

        # Add reason for non-completion if applicable
        if not completed:
            if not phase_config.enabled:
                metadata["reason"] = "disabled"
            else:
                metadata["reason"] = "not_run"

        self._phase_metadata.append(metadata)

    def _save_metadata(self, completed_phases: List[LlmPhase]) -> None:
        """
        Save comprehensive metadata about the pipeline run to the output directory.

        This method creates a comprehensive JSON file containing metadata about
        the entire pipeline run, including information about all phases (both
        completed and not run), their configurations, execution status, and
        system prompt information.

        Args:
            completed_phases (List[LlmPhase]): List of phases that were successfully completed
        """
        metadata = {
            "metadata_version": METADATA_VERSION,
            "run_timestamp": datetime.now().isoformat(),
            "book_name": self.config.book_name,
            "author_name": self.config.author_name,
            "input_file": str(self.config.input_file),
            "original_file": str(self.config.original_file),
            "output_directory": str(self.config.output_dir),
            "length_reduction": self.config.length_reduction,
            "phases": self._phase_metadata,
        }

        # Save metadata to output directory
        metadata_file = self.config.output_dir / f"pipeline_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(file=metadata_file, mode="w", encoding="utf-8") as f:
                json.dump(obj=metadata, fp=f, indent=2, ensure_ascii=False)
            logger.info(f"Pipeline metadata saved to: {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save pipeline metadata: {str(e)}")

    def _get_phase_output_path(self, phase_index: int) -> Path:
        """
        Generate output path for a phase.

        This method determines the output file path for a specific phase,
        either using a custom path if specified in the configuration or
        generating a default path based on the phase type and index.

        Args:
            phase_index (int): The index of the phase in the pipeline sequence

        Returns:
            Path: The output file path for the phase
        """
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
        """
        Determine input path for a phase based on the previous phase's output.

        The first phase uses the run's input file, while subsequent phases
        use the output from the previous phase as their input.

        Args:
            phase_index (int): The index of the phase in the pipeline sequence

        Returns:
            Path: The input file path for the phase
        """
        # First phase uses the run's input file
        if phase_index == 0:
            return self.config.input_file

        # Subsequent phases use the previous phase's output
        previous_phase_index = phase_index - 1
        return self._get_phase_output_path(phase_index=previous_phase_index)

    def _initialize_phase(self, phase_index: int) -> Optional[LlmPhase]:
        """
        Initialize a single phase when it's about to run.

        This method creates and configures a phase instance based on the
        configuration, including setting up the LLM model, determining
        input/output paths, and creating the appropriate phase type.

        Args:
            phase_index (int): The index of the phase to initialize

        Returns:
            Optional[LlmPhase]: The initialized phase instance, or None if the phase is disabled
        """
        phase_config = self.config.phases[phase_index]
        if not phase_config.enabled:
            logger.info(f"Skipping disabled phase: {phase_config.phase_type.name}")
            return None

        input_path = self._get_phase_input_path(phase_index=phase_index)
        output_path = self._get_phase_output_path(phase_index=phase_index)

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

        # Create PhaseConfig for the factory (do NOT include length_reduction)
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

        # Create the phase instance using the factory, passing length_reduction as a kwarg
        phase_factory_kwargs = {"length_reduction": self.config.length_reduction}
        if phase_config.phase_type in [PhaseType.MODERNIZE, PhaseType.EDIT, PhaseType.FINAL, PhaseType.ANNOTATE]:
            phase = PhaseFactory.create_standard_phase(config=factory_config, **phase_factory_kwargs)
        elif phase_config.phase_type == PhaseType.INTRODUCTION:
            phase = PhaseFactory.create_introduction_annotation_phase(config=factory_config, **phase_factory_kwargs)
        elif phase_config.phase_type == PhaseType.SUMMARY:
            phase = PhaseFactory.create_summary_annotation_phase(config=factory_config, **phase_factory_kwargs)
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
        """
        Run all enabled phases in order.

        This method executes all enabled phases in the configured sequence,
        handling phase initialization, execution, error handling, and metadata
        collection. It ensures proper cleanup and metadata saving even if
        errors occur.

        Args:
            **kwargs: Additional arguments to pass to the processing methods
        """
        phase_order = self.config.get_phase_order()
        logger.info(f"Starting pipeline with phases: {[p.name for p in phase_order]}")

        completed_phases: List[LlmPhase] = []

        try:
            for i, phase_config in enumerate(self.config.phases):
                if not phase_config.enabled:
                    logger.info(f"Skipping disabled phase: {phase_config.phase_type.name}")
                    # Collect metadata for disabled phases
                    self._collect_phase_metadata(phase=None, phase_index=i, completed=False)
                    continue

                logger.info(f"Proceeding with phase: {phase_config.phase_type.name}")

                phase = self._initialize_phase(phase_index=i)
                if not phase:
                    logger.warning(f"Could not initialize phase: {phase_config.phase_type.name}")
                    # Collect metadata for phases that couldn't be initialized
                    self._collect_phase_metadata(phase=None, phase_index=i, completed=False)
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
                    # Collect metadata for completed phases
                    self._collect_phase_metadata(phase=phase, phase_index=i, completed=True)
                except LlmModelError as e:
                    logger.error(f"LLM model error in phase {phase.name}: {str(e)}")
                    logger.error("Pipeline stopped due to LLM model failure after max retries")
                    # Collect metadata for failed phases
                    self._collect_phase_metadata(phase=phase, phase_index=i, completed=False)
                    raise
                except Exception as e:
                    logger.error(f"Error in phase {phase.name}: {str(e)}", exc_info=True)
                    # Collect metadata for failed phases
                    self._collect_phase_metadata(phase=phase, phase_index=i, completed=False)
                    raise

            logger.success("Pipeline completed successfully")
        finally:
            # Save comprehensive metadata about the run (whether successful or failed)
            self._save_metadata(completed_phases=completed_phases)


def run_pipeline(config: RunConfig) -> None:
    """
    Run the pipeline with the given configuration.

    This is the main entry point for running the LLM processing pipeline.
    It creates a Pipeline instance and executes it with the provided configuration.

    Args:
        config (RunConfig): Configuration object containing all run parameters

    Raises:
        Exception: If the pipeline fails to execute
    """
    logger.info(f"Running pipeline for {config.book_name}")

    try:
        pipeline = Pipeline(config=config)
        pipeline.run()
        logger.success("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
