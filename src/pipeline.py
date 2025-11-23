import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.llm_model import ModelConfig

import requests
from loguru import logger

from src.config import PhaseType, RunConfig
from src.constants import INPUT_FILE_INDEX_PREFIX, OPENROUTER_API_TIMEOUT
from src.cost_tracking_wrapper import calculate_and_log_costs
from src.llm_model import LlmModel, LlmModelError
from src.llm_phase import LlmPhase
from src.logging_config import setup_logging
from src.phase_factory import PhaseFactory

# Initialize module-level logger
module_logger = setup_logging(log_name="pipeline")

# Metadata version for compatibility
METADATA_VERSION = "0.0.0-alpha"


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
        self._model_cache: Dict[str, Dict[str, str]] = {}  # Cache for model information
        self._llm_model_instances: Dict[str, LlmModel] = {}  # Cache for LlmModel instances (connection pooling)

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

    def _get_or_create_model(self, model_config: "ModelConfig", temperature: float) -> LlmModel:
        """
        Get or create a cached LlmModel instance for connection pooling.

        This method implements model instance caching to reuse connection pools
        across phases. Models are cached based on their configuration to ensure
        connection pool reuse when the same model is used in multiple phases.

        Args:
            model_config: Configuration for the model to create
            temperature: Temperature setting for the model

        Returns:
            LlmModel: A cached or newly created LlmModel instance
        """
        # Create cache key from model config and temperature
        # Include temperature in key as it affects model behavior
        cache_key = f"{model_config.provider.value}:{model_config.model_id}:{temperature}"

        # Return cached instance if available
        if cache_key in self._llm_model_instances:
            logger.debug(f"Reusing cached LlmModel instance: {cache_key}")
            return self._llm_model_instances[cache_key]

        # Create new instance and cache it
        logger.debug(f"Creating new LlmModel instance: {cache_key}")
        model = LlmModel.create(
            model=model_config,
            temperature=temperature,
        )
        self._llm_model_instances[cache_key] = model
        return model

    def _cleanup_model_instances(self) -> None:
        """
        Clean up all cached model instances and their connection pools.

        This method should be called at the end of the pipeline run to properly
        close all connection pools and release resources.
        """
        logger.debug(f"Cleaning up {len(self._llm_model_instances)} cached model instances")
        for cache_key, model in self._llm_model_instances.items():
            try:
                # Close OpenRouter client session if it exists
                if hasattr(model, "_client") and hasattr(model._client, "close"):
                    model._client.close()
                    logger.debug(f"Closed connection pool for model: {cache_key}")
            except Exception as e:
                # Cleanup errors are non-critical but should be logged
                logger.debug(f"Error closing model instance {cache_key}: {e}")

    def _copy_input_file_to_output(self) -> None:
        """
        Copy the input file to the output directory with index "00".

        This method copies the original input file to the output directory
        with a filename that starts with the input file index prefix to maintain
        the proper ordering of files in the pipeline output.
        """
        input_filename = self.config.input_file.name
        output_filename = f"{INPUT_FILE_INDEX_PREFIX}-{input_filename}"
        output_path = self.config.output_dir / output_filename

        try:
            shutil.copy2(src=self.config.input_file, dst=output_path)
            logger.info(f"Copied input file to output directory: {output_path}")
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"Failed to copy input file to output directory: {str(e)}")
            raise

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
                if isinstance(processor, str):
                    post_processors_info.append(processor)
                elif hasattr(processor, "name"):
                    post_processors_info.append(processor.name)
                else:
                    post_processors_info.append(str(processor))

        # Get model metadata
        model_metadata = self._get_model_metadata(model_type=phase_config.model)

        # Base metadata that's always available
        metadata = {
            "phase_name": phase.name if phase else phase_config.phase_type.name.lower(),
            "phase_index": phase_index,
            "phase_type": phase_config.phase_type.name,
            "enabled": phase_config.enabled,
            "temperature": phase_config.temperature,
            "post_processors": post_processors_info,
            "post_processor_count": len(post_processors_info),
            "completed": completed,
            "book_id": self.config.book_id,
            "book_name": self.config.book_name,
            "author_name": self.config.author_name,
        }

        # Add model metadata
        metadata.update(model_metadata)

        # Add max_workers to metadata
        metadata["max_workers"] = getattr(self.config, "max_workers", None)

        # Add phase-specific information if phase exists
        if phase:
            metadata.update(
                {
                    "input_file": phase.input_file_path.as_posix(),
                    "output_file": phase.output_file_path.as_posix(),
                    "system_prompt_path": phase.system_prompt_path.as_posix() if phase.system_prompt_path else None,
                    "user_prompt_path": phase.user_prompt_path.as_posix() if phase.user_prompt_path else None,
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
                    "system_prompt_path": phase_config.system_prompt_path.as_posix()
                    if phase_config.system_prompt_path
                    else None,
                    "user_prompt_path": phase_config.user_prompt_path.as_posix()
                    if phase_config.user_prompt_path
                    else None,
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
            "book_id": self.config.book_id,
            "book_name": self.config.book_name,
            "author_name": self.config.author_name,
            "input_file": self.config.input_file.as_posix(),
            "original_file": self.config.original_file.as_posix(),
            "output_directory": self.config.output_dir.as_posix(),
            "length_reduction": self.config.length_reduction,
            "phases": self._phase_metadata,
        }

        # Save metadata to output directory
        metadata_file = self.config.output_dir / f"pipeline_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(file=metadata_file, mode="w", encoding="utf-8") as f:
                json.dump(obj=metadata, fp=f, indent=2, ensure_ascii=False)
            logger.info(f"Pipeline metadata saved to: {metadata_file}")
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"Failed to save pipeline metadata: {str(e)}")
            logger.exception("Metadata save error details")

    def _save_cost_analysis(self, cost_analysis: Dict[str, Any]) -> None:
        """
        Save cost analysis data to a separate JSON file in the output directory.

        Args:
            cost_analysis (Dict[str, Any]): Cost analysis data from the run
        """
        # Save cost analysis to output directory
        cost_file = self.config.output_dir / f"cost_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(file=cost_file, mode="w", encoding="utf-8") as f:
                json.dump(obj=cost_analysis, fp=f, indent=2, ensure_ascii=False)
            logger.info(f"Cost analysis saved to: {cost_file}")
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"Failed to save cost analysis: {str(e)}")
            logger.exception("Cost analysis save error details")

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

        # Get or create the model (uses caching for connection pool reuse)
        model = self._get_or_create_model(
            model_config=phase_config.model,
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
            llm_model_instance=model,
            temperature=phase_config.temperature,
            reasoning=phase_config.reasoning,
            post_processors=phase_config.post_processors,
            use_batch=phase_config.use_batch,
            batch_size=phase_config.batch_size,
        )

        # Create the phase instance using the factory with explicit arguments
        phase: LlmPhase
        if phase_config.phase_type in [PhaseType.MODERNIZE, PhaseType.EDIT, PhaseType.FINAL, PhaseType.ANNOTATE]:
            phase = PhaseFactory.create_standard_phase(
                config=factory_config,
                length_reduction=self.config.length_reduction,
                tags_to_preserve=self.config.tags_to_preserve,
                max_workers=self.config.max_workers,
            )
        elif phase_config.phase_type == PhaseType.INTRODUCTION:
            phase = PhaseFactory.create_introduction_annotation_phase(
                config=factory_config,
                length_reduction=self.config.length_reduction,
                tags_to_preserve=self.config.tags_to_preserve,
                max_workers=self.config.max_workers,
            )
        elif phase_config.phase_type == PhaseType.SUMMARY:
            phase = PhaseFactory.create_summary_annotation_phase(
                config=factory_config,
                length_reduction=self.config.length_reduction,
                tags_to_preserve=self.config.tags_to_preserve,
                max_workers=self.config.max_workers,
            )
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

    def _fetch_model_info(self, model_id: str) -> Optional[Dict[str, str]]:
        """
        Fetch model information from OpenRouter API.

        This method retrieves model information from the OpenRouter API
        to get the clean name and other details for a given model ID.

        Args:
            model_id (str): The model ID to look up

        Returns:
            Optional[Dict[str, str]]: Model information with 'id' and 'name' keys,
                or None if the model is not found or API call fails
        """
        # Check cache first
        if model_id in self._model_cache:
            return self._model_cache[model_id]

        try:
            url = "https://openrouter.ai/api/v1/models"
            response = requests.get(url=url, timeout=OPENROUTER_API_TIMEOUT)
            response.raise_for_status()

            data = response.json()
            models = data.get("data", [])

            # Find the model by ID
            for model in models:
                if model.get("id") == model_id:
                    model_info = {"id": model["id"], "name": model["name"]}
                    # Cache the result
                    self._model_cache[model_id] = model_info
                    return model_info

            # Model not found
            logger.warning(f"Model not found in OpenRouter API: {model_id}")
            return None

        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
            # Model info is optional metadata; graceful degradation on failure
            logger.warning(f"Failed to fetch model info for {model_id}: {str(e)}")
            return None

    def _get_model_metadata(self, model_type: "ModelConfig") -> Dict[str, Any]:
        """
        Get model metadata from ModelConfig object.

        Args:
            model_type: The model configuration (ModelConfig)

        Returns:
            Dict[str, Any]: Model metadata with provider and model information
        """
        model_id = model_type.model_id
        provider = model_type.provider.value
        provider_model_name = model_type.provider_model_name

        # For non-OpenRouter providers, don't try to fetch from OpenRouter API
        if provider != "openrouter":
            return {
                "model_type": model_id,
                "provider": provider,
                "provider_model_name": provider_model_name,
                "model": {"id": model_id, "name": provider_model_name or model_id},
            }

        # Try to fetch info from OpenRouter API for OpenRouter models
        model_info = self._fetch_model_info(model_id=model_id)

        if model_info:
            return {
                "model_type": model_id,
                "provider": provider,
                "provider_model_name": provider_model_name,
                "model": model_info,
            }
        else:
            # Fallback if we can't fetch clean info from OpenRouter
            return {
                "model_type": model_id,
                "provider": provider,
                "provider_model_name": provider_model_name,
                "model": {"id": model_id, "name": provider_model_name or model_id},
            }

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

        # Copy input file to output directory with index "00"
        self._copy_input_file_to_output()

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
            # Calculate and log costs at the end of the run
            cost_analysis = None
            phase_names = [phase.name for phase in completed_phases]
            if phase_names:
                run_costs = calculate_and_log_costs(phase_names)
                if run_costs:
                    # Convert RunCosts to dictionary for JSON serialization
                    cost_analysis = {
                        "total_phases": run_costs.total_phases,
                        "completed_phases": run_costs.completed_phases,
                        "total_generations": run_costs.total_generations,
                        "total_prompt_tokens": run_costs.total_prompt_tokens,
                        "total_completion_tokens": run_costs.total_completion_tokens,
                        "total_tokens": run_costs.total_tokens,
                        "total_cost": run_costs.total_cost,
                        "currency": run_costs.currency,
                        "phase_costs": [
                            {
                                "phase_name": phase.phase_name,
                                "phase_index": phase.phase_index,
                                "generation_ids": phase.generation_ids,
                                "total_prompt_tokens": phase.total_prompt_tokens,
                                "total_completion_tokens": phase.total_completion_tokens,
                                "total_tokens": phase.total_tokens,
                                "total_cost": phase.total_cost,
                                "currency": phase.currency,
                                "generation_count": phase.generation_count,
                            }
                            for phase in run_costs.phase_costs
                        ],
                    }

            # Save comprehensive metadata about the run (whether successful or failed)
            self._save_metadata(completed_phases=completed_phases)

            # Save cost analysis data if available
            if cost_analysis:
                self._save_cost_analysis(cost_analysis=cost_analysis)

            # Clean up model instances and connection pools
            self._cleanup_model_instances()


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
