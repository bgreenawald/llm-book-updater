from pathlib import Path
from typing import Any, List, Optional, TypedDict, Union

from src.api.config import PhaseConfig, PhaseType, PostProcessorType
from src.models.model import LlmModel
from src.phases.annotation import IntroductionAnnotationPhase, SummaryAnnotationPhase
from src.phases.standard import StandardLlmPhase
from src.phases.two_stage import StageConfig, TwoStageFinalPhase
from src.phases.utils import read_file
from src.processing.post_processors import (
    EnsureBlankLineProcessor,
    NoNewHeadersPostProcessor,
    OrderQuoteAnnotationProcessor,
    PostProcessor,
    PostProcessorChain,
    PreserveFStringTagsProcessor,
    RemoveBlankLinesInListProcessor,
    RemoveMarkdownBlocksProcessor,
    RemoveTrailingWhitespaceProcessor,
    RemoveXmlTagsProcessor,
    RevertRemovedBlockLines,
    ValidateNonEmptySectionProcessor,
)


class SafeDict(dict):
    """Dictionary subclass that preserves unknown placeholders during format_map.

    When a key is missing, __missing__ returns the original placeholder format
    (e.g., "{key}") instead of raising a KeyError. This enables safe partial
    formatting where unknown placeholders are preserved in the output.
    """

    def __missing__(self, key: str) -> str:
        """Return the original placeholder format for missing keys.

        Args:
            key: The missing key name

        Returns:
            The original placeholder format string (e.g., "{key}")
        """
        return "{%s}" % key


class ValidatedPhaseFields(TypedDict):
    """Type-safe container for validated phase fields."""

    name: str
    input_file_path: Path
    output_file_path: Path
    original_file_path: Path
    book_name: str
    author_name: str
    llm_model_instance: LlmModel


class PhaseFactory:
    """
    Factory class for creating different types of LLM phases with appropriate
    post-processor configurations.

    This allows for easy creation of phases with different post-processing needs
    without having to manually configure post-processor chains for each phase type.
    """

    DEFAULT_POST_PROCESSORS: dict[PhaseType, list[PostProcessorType]] = {
        PhaseType.MODERNIZE: [
            PostProcessorType.VALIDATE_NON_EMPTY_SECTION,
            PostProcessorType.NO_NEW_HEADERS,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE,
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.REMOVE_MARKDOWN_BLOCKS,
            PostProcessorType.PRESERVE_F_STRING_TAGS,
            PostProcessorType.ENSURE_BLANK_LINE,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST,
        ],
        PhaseType.EDIT: [
            PostProcessorType.VALIDATE_NON_EMPTY_SECTION,
            PostProcessorType.NO_NEW_HEADERS,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE,
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.REMOVE_MARKDOWN_BLOCKS,
            PostProcessorType.PRESERVE_F_STRING_TAGS,
            PostProcessorType.ENSURE_BLANK_LINE,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST,
        ],
        PhaseType.FINAL: [
            PostProcessorType.VALIDATE_NON_EMPTY_SECTION,
            PostProcessorType.NO_NEW_HEADERS,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE,
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.REMOVE_MARKDOWN_BLOCKS,
            PostProcessorType.PRESERVE_F_STRING_TAGS,
            PostProcessorType.ENSURE_BLANK_LINE,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST,
        ],
        PhaseType.FINAL_TWO_STAGE: [
            # Same as FINAL - post-processing applies to IMPLEMENT output only
            PostProcessorType.VALIDATE_NON_EMPTY_SECTION,
            PostProcessorType.NO_NEW_HEADERS,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE,
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.REMOVE_MARKDOWN_BLOCKS,
            PostProcessorType.PRESERVE_F_STRING_TAGS,
            PostProcessorType.ENSURE_BLANK_LINE,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST,
        ],
        PhaseType.INTRODUCTION: [
            PostProcessorType.VALIDATE_NON_EMPTY_SECTION,
            PostProcessorType.NO_NEW_HEADERS,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE,
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.REMOVE_MARKDOWN_BLOCKS,
            PostProcessorType.PRESERVE_F_STRING_TAGS,
            PostProcessorType.ENSURE_BLANK_LINE,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST,
        ],
        PhaseType.SUMMARY: [
            PostProcessorType.VALIDATE_NON_EMPTY_SECTION,
            PostProcessorType.REVERT_REMOVED_BLOCK_LINES,
            PostProcessorType.NO_NEW_HEADERS,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE,
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.REMOVE_MARKDOWN_BLOCKS,
            PostProcessorType.PRESERVE_F_STRING_TAGS,
            PostProcessorType.ENSURE_BLANK_LINE,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST,
        ],
        PhaseType.ANNOTATE: [
            PostProcessorType.VALIDATE_NON_EMPTY_SECTION,
            PostProcessorType.REVERT_REMOVED_BLOCK_LINES,
            PostProcessorType.ORDER_QUOTE_ANNOTATION,
            PostProcessorType.NO_NEW_HEADERS,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE,
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.REMOVE_MARKDOWN_BLOCKS,
            PostProcessorType.PRESERVE_F_STRING_TAGS,
            PostProcessorType.ENSURE_BLANK_LINE,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST,
        ],
    }

    @staticmethod
    def _validate_required_phase_fields(config: PhaseConfig, phase_type: str) -> ValidatedPhaseFields:
        """
        Validate that all required fields are present in the config.

        Args:
            config (PhaseConfig): Configuration object to validate
            phase_type (str): Name of the phase type for error messages

        Returns:
            ValidatedPhaseFields: Type-safe container with validated non-None fields

        Raises:
            ValueError: If any required field is None
        """
        if config.name is None:
            raise ValueError(f"name is required for {phase_type}")
        if config.input_file_path is None:
            raise ValueError(f"input_file_path is required for {phase_type}")
        if config.output_file_path is None:
            raise ValueError(f"output_file_path is required for {phase_type}")
        if config.original_file_path is None:
            raise ValueError(f"original_file_path is required for {phase_type}")
        if config.book_name is None:
            raise ValueError(f"book_name is required for {phase_type}")
        if config.author_name is None:
            raise ValueError(f"author_name is required for {phase_type}")
        if config.llm_model_instance is None:
            raise ValueError(f"llm_model_instance is required for {phase_type}")

        return ValidatedPhaseFields(
            name=config.name,
            input_file_path=config.input_file_path,
            output_file_path=config.output_file_path,
            original_file_path=config.original_file_path,
            book_name=config.book_name,
            author_name=config.author_name,
            llm_model_instance=config.llm_model_instance,
        )

    @staticmethod
    def _prepare_phase_parameters(
        config: PhaseConfig,
        phase_type: str,
        tags_to_preserve: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Prepare common parameters for phase instantiation.

        This method handles:
        - Creating the post-processor chain
        - Validating required fields
        - Constructing the parameter dictionary

        Args:
            config (PhaseConfig): Configuration object containing all phase parameters
            phase_type (str): Name of the phase type for error messages
            tags_to_preserve: Tags to preserve during processing
            max_workers: Maximum number of workers for parallel processing

        Returns:
            dict[str, Any]: Dictionary of parameters ready for phase instantiation
        """
        post_processor_chain = PhaseFactory._create_post_processor_chain(
            post_processors=config.post_processors, phase_type=config.phase_type, tags_to_preserve=tags_to_preserve
        )

        validated = PhaseFactory._validate_required_phase_fields(config=config, phase_type=phase_type)

        return {
            "name": validated["name"],
            "input_file_path": validated["input_file_path"],
            "output_file_path": validated["output_file_path"],
            "original_file_path": validated["original_file_path"],
            "system_prompt_path": config.system_prompt_path,
            "user_prompt_path": config.user_prompt_path,
            "book_name": validated["book_name"],
            "author_name": validated["author_name"],
            "model": validated["llm_model_instance"],
            "max_workers": max_workers,
            "reasoning": config.reasoning,
            "llm_kwargs": config.llm_kwargs,
            "post_processor_chain": post_processor_chain,
            "use_batch": config.use_batch,
            "batch_size": config.batch_size,
            "enable_retry": config.enable_retry,
            "max_retries": config.max_retries,
            "use_subblocks": config.use_subblocks,
            "max_subblock_tokens": config.max_subblock_tokens,
            "min_subblock_tokens": config.min_subblock_tokens,
            "skip_if_less_than_tokens": config.skip_if_less_than_tokens,
        }

    @staticmethod
    def create_standard_phase(
        config: PhaseConfig,
        tags_to_preserve: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
    ) -> StandardLlmPhase:
        """
        Create a standard LLM phase with optional post-processors.

        Args:
            config (PhaseConfig): Configuration object containing all phase parameters
            tags_to_preserve: Tags to preserve during processing
            max_workers: Maximum number of workers for parallel processing

        Returns:
            StandardLlmPhase: Configured standard phase
        """
        params = PhaseFactory._prepare_phase_parameters(
            config=config,
            phase_type="StandardLlmPhase",
            tags_to_preserve=tags_to_preserve,
            max_workers=max_workers,
        )
        return StandardLlmPhase(**params)

    @staticmethod
    def create_introduction_annotation_phase(
        config: PhaseConfig,
        tags_to_preserve: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
    ) -> IntroductionAnnotationPhase:
        """
        Create an introduction annotation phase with optional post-processors.

        Args:
            config (PhaseConfig): Configuration object containing all phase parameters
            tags_to_preserve: Tags to preserve during processing
            max_workers: Maximum number of workers for parallel processing

        Returns:
            IntroductionAnnotationPhase: Configured introduction annotation phase
        """
        params = PhaseFactory._prepare_phase_parameters(
            config=config,
            phase_type="IntroductionAnnotationPhase",
            tags_to_preserve=tags_to_preserve,
            max_workers=max_workers,
        )
        return IntroductionAnnotationPhase(**params)

    @staticmethod
    def create_summary_annotation_phase(
        config: PhaseConfig,
        tags_to_preserve: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
    ) -> SummaryAnnotationPhase:
        """
        Create a summary annotation phase with optional post-processors.

        Args:
            config (PhaseConfig): Configuration object containing all phase parameters
            tags_to_preserve: Tags to preserve during processing
            max_workers: Maximum number of workers for parallel processing

        Returns:
            SummaryAnnotationPhase: Configured summary annotation phase
        """
        params = PhaseFactory._prepare_phase_parameters(
            config=config,
            phase_type="SummaryAnnotationPhase",
            tags_to_preserve=tags_to_preserve,
            max_workers=max_workers,
        )
        return SummaryAnnotationPhase(**params)

    @staticmethod
    def create_two_stage_final_phase(
        config: PhaseConfig,
        identify_model: LlmModel,
        implement_model: LlmModel,
        tags_to_preserve: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
    ) -> TwoStageFinalPhase:
        """
        Create a two-stage FINAL phase with separate models for identify/implement.

        This phase runs two internal stages:
        1. IDENTIFY: Analyzes the passage and identifies refinement opportunities
        2. IMPLEMENT: Applies the identified changes

        Args:
            config: Configuration object. Must have two_stage_config set.
            identify_model: LLM model instance for the IDENTIFY stage.
            implement_model: LLM model instance for the IMPLEMENT stage.
            tags_to_preserve: Tags to preserve during processing.
            max_workers: Maximum number of workers for parallel processing.

        Returns:
            TwoStageFinalPhase: Configured two-stage phase.

        Raises:
            ValueError: If required configuration fields are missing.
        """
        if config.two_stage_config is None:
            raise ValueError("two_stage_config is required for FINAL_TWO_STAGE phase")

        # Validate required fields
        if config.name is None:
            raise ValueError("name is required for TwoStageFinalPhase")
        if config.input_file_path is None:
            raise ValueError("input_file_path is required for TwoStageFinalPhase")
        if config.output_file_path is None:
            raise ValueError("output_file_path is required for TwoStageFinalPhase")
        if config.original_file_path is None:
            raise ValueError("original_file_path is required for TwoStageFinalPhase")
        if config.book_name is None:
            raise ValueError("book_name is required for TwoStageFinalPhase")
        if config.author_name is None:
            raise ValueError("author_name is required for TwoStageFinalPhase")

        # Load and format prompts
        identify_system = PhaseFactory._load_and_format_prompt(
            Path("./prompts/final_identify_system.md"),
            tags_to_preserve,
        )
        identify_user = read_file(Path("./prompts/final_identify_user.md"))
        implement_system = PhaseFactory._load_and_format_prompt(
            Path("./prompts/final_implement_system.md"),
            tags_to_preserve,
        )
        implement_user = read_file(Path("./prompts/final_implement_user.md"))

        # Create stage configs
        identify_stage = StageConfig(
            model=identify_model,
            system_prompt=identify_system,
            user_prompt_template=identify_user,
            reasoning=config.two_stage_config.identify_reasoning,
        )

        implement_stage = StageConfig(
            model=implement_model,
            system_prompt=implement_system,
            user_prompt_template=implement_user,
        )

        # Create post-processor chain (applies to IMPLEMENT output only)
        post_processor_chain = PhaseFactory._create_post_processor_chain(
            post_processors=config.post_processors,
            phase_type=config.phase_type,
            tags_to_preserve=tags_to_preserve,
        )

        return TwoStageFinalPhase(
            name=config.name,
            input_file_path=config.input_file_path,
            output_file_path=config.output_file_path,
            original_file_path=config.original_file_path,
            book_name=config.book_name,
            author_name=config.author_name,
            identify_config=identify_stage,
            implement_config=implement_stage,
            post_processor_chain=post_processor_chain,
            max_workers=max_workers,
            llm_kwargs=config.llm_kwargs,
            use_batch=config.use_batch,
            batch_size=config.batch_size,
            enable_retry=config.enable_retry,
            max_retries=config.max_retries,
            skip_if_less_than_tokens=config.skip_if_less_than_tokens,
            tags_to_preserve=tags_to_preserve,
        )

    @staticmethod
    def _load_and_format_prompt(path: Path, tags_to_preserve: Optional[List[str]]) -> str:
        """Load a prompt file and format with tags.

        Args:
            path: Path to the prompt file
            tags_to_preserve: Tags to preserve for formatting

        Returns:
            Formatted prompt content
        """

        content = read_file(path)

        # Format with tags_to_preserve using safe partial formatting
        if tags_to_preserve:
            format_params = {}
            for tag in tags_to_preserve:
                tag_name = tag.strip("{}")
                format_params[tag_name] = tag
            content = content.format_map(SafeDict(format_params))

        return content

    @staticmethod
    def _create_built_in_processor(processor_name: str) -> Optional[PostProcessor]:
        """
        Create a built-in post-processor by name.

        Args:
            processor_name (str): Name of the built-in processor

        Returns:
            Optional[PostProcessor]: The created processor or None if not found
        """
        from typing import Type

        processors: dict[str, Type[PostProcessor]] = {
            "ensure_blank_line": EnsureBlankLineProcessor,
            "remove_xml_tags": RemoveXmlTagsProcessor,
            "remove_trailing_whitespace": RemoveTrailingWhitespaceProcessor,
            "remove_markdown_blocks": RemoveMarkdownBlocksProcessor,
            "order_quote_annotation": OrderQuoteAnnotationProcessor,
            "no_new_headers": NoNewHeadersPostProcessor,
            "revert_removed_block_lines": RevertRemovedBlockLines,
            "remove_blank_lines_in_list": RemoveBlankLinesInListProcessor,
            "preserve_fstring_tags": PreserveFStringTagsProcessor,
            "validate_non_empty_section": ValidateNonEmptySectionProcessor,
        }

        processor_class = processors.get(processor_name.lower())
        if processor_class is not None:
            return processor_class()  # type: ignore[abstract, call-arg]

        return None

    @staticmethod
    def _create_processor_from_enum(
        processor_type: PostProcessorType, tags_to_preserve: Optional[List[str]] = None
    ) -> Optional[PostProcessor]:
        """
        Create a built-in post-processor from PostProcessorType enum.

        Args:
            processor_type (PostProcessorType): The post-processor type
            tags_to_preserve (Optional[List[str]]): Tags to preserve for PreserveFStringTagsProcessor

        Returns:
            Optional[PostProcessor]: The created processor or None if not found
        """
        from typing import Type

        processor_mapping: dict[PostProcessorType, Type[PostProcessor]] = {
            PostProcessorType.ENSURE_BLANK_LINE: EnsureBlankLineProcessor,
            PostProcessorType.REMOVE_XML_TAGS: RemoveXmlTagsProcessor,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE: RemoveTrailingWhitespaceProcessor,
            PostProcessorType.REMOVE_MARKDOWN_BLOCKS: RemoveMarkdownBlocksProcessor,
            PostProcessorType.ORDER_QUOTE_ANNOTATION: OrderQuoteAnnotationProcessor,
            PostProcessorType.NO_NEW_HEADERS: NoNewHeadersPostProcessor,
            PostProcessorType.REVERT_REMOVED_BLOCK_LINES: RevertRemovedBlockLines,
            PostProcessorType.PRESERVE_F_STRING_TAGS: PreserveFStringTagsProcessor,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST: RemoveBlankLinesInListProcessor,
            PostProcessorType.VALIDATE_NON_EMPTY_SECTION: ValidateNonEmptySectionProcessor,
        }

        processor_class = processor_mapping.get(processor_type)
        if processor_class is not None:
            if processor_type == PostProcessorType.PRESERVE_F_STRING_TAGS and tags_to_preserve:
                return processor_class(config={"tags_to_preserve": tags_to_preserve})  # type: ignore[abstract, call-arg]
            else:
                return processor_class()  # type: ignore[abstract, call-arg]

        return None

    @staticmethod
    def _create_formatting_processor_chain() -> PostProcessorChain:
        """
        Create a formatting processor chain that combines multiple formatting processors.

        This is a convenience method that creates a chain with commonly used
        formatting processors in the correct order.

        Returns:
            PostProcessorChain: Chain with formatting processors
        """
        chain = PostProcessorChain()
        chain.add_processor(processor=RemoveXmlTagsProcessor())
        chain.add_processor(processor=RemoveTrailingWhitespaceProcessor())
        chain.add_processor(processor=EnsureBlankLineProcessor())
        chain.add_processor(processor=OrderQuoteAnnotationProcessor())
        return chain

    @staticmethod
    def _create_post_processor_chain(
        post_processors: Optional[List[Union[str, PostProcessor, PostProcessorType]]] = None,
        phase_type: Optional[PhaseType] = None,
        tags_to_preserve: Optional[List[str]] = None,
    ) -> Optional[PostProcessorChain]:
        """
        Create a post-processor chain from a unified list of post-processors.

        The list can contain:
        - Strings: Names of built-in post-processors (e.g., "ensure_blank_line")
        - PostProcessor instances: Custom post-processor objects
        - PostProcessorType enum values: Type-safe post-processor types
        - Special aliases: "formatting" for a predefined chain of formatting processors

        If no post_processors are provided and a phase_type is specified,
        the default post-processors for that phase type will be used.

        Args:
            post_processors (Optional[List[Union[str, PostProcessor, PostProcessorType]]]):
                Unified list of post-processors (strings, instances, or enum values)
            phase_type (Optional[PhaseType]): Phase type for default post-processors
            tags_to_preserve (Optional[List[str]]): Tags to preserve for PreserveFStringTagsProcessor

        Returns:
            Optional[PostProcessorChain]: Configured post-processor chain or None
        """
        # Use default post-processors if none provided and phase_type is specified
        if not post_processors and phase_type:
            default_processors = PhaseFactory.DEFAULT_POST_PROCESSORS.get(phase_type, [])
            post_processors = list(default_processors)

        if not post_processors:
            return None

        chain = PostProcessorChain()

        for processor_item in post_processors:
            if isinstance(processor_item, str):
                # Handle string-based processors (built-in or aliases)
                if processor_item.lower() == "formatting":
                    # Handle special "formatting" alias
                    formatting_chain = PhaseFactory._create_formatting_processor_chain()
                    # Add all processors from the formatting chain
                    for processor in formatting_chain.processors:
                        chain.add_processor(processor=processor)
                else:
                    # Handle built-in processor by name
                    built_in_processor = PhaseFactory._create_built_in_processor(processor_name=processor_item)
                    if built_in_processor is not None:
                        chain.add_processor(processor=built_in_processor)
            elif isinstance(processor_item, PostProcessor):
                # Handle custom PostProcessor instances
                chain.add_processor(processor=processor_item)
            elif isinstance(processor_item, PostProcessorType):
                # Handle PostProcessorType enum values
                enum_processor = PhaseFactory._create_processor_from_enum(
                    processor_type=processor_item, tags_to_preserve=tags_to_preserve
                )
                if enum_processor is not None:
                    chain.add_processor(processor=enum_processor)
            else:
                # Skip invalid items
                continue

        return chain
