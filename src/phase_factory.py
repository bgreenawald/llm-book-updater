from typing import Any, List, Optional, Union

from src.config import PhaseConfig, PhaseType, PostProcessorType
from src.llm_phase import IntroductionAnnotationPhase, StandardLlmPhase, SummaryAnnotationPhase
from src.post_processors import (
    EnsureBlankLineProcessor,
    NoNewHeadersPostProcessor,
    OrderQuoteAnnotationProcessor,
    PostProcessor,
    PostProcessorChain,
    PreserveFStringTagsProcessor,
    RemoveBlankLinesInListProcessor,
    RemoveTrailingWhitespaceProcessor,
    RemoveXmlTagsProcessor,
    RevertRemovedBlockLines,
)


class PhaseFactory:
    """
    Factory class for creating different types of LLM phases with appropriate
    post-processor configurations.

    This allows for easy creation of phases with different post-processing needs
    without having to manually configure post-processor chains for each phase type.
    """

    DEFAULT_POST_PROCESSORS: dict[PhaseType, list[PostProcessorType]] = {
        PhaseType.MODERNIZE: [
            PostProcessorType.NO_NEW_HEADERS,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE,
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.PRESERVE_F_STRING_TAGS,
            PostProcessorType.ENSURE_BLANK_LINE,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST,
        ],
        PhaseType.EDIT: [
            PostProcessorType.NO_NEW_HEADERS,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE,
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.PRESERVE_F_STRING_TAGS,
            PostProcessorType.ENSURE_BLANK_LINE,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST,
        ],
        PhaseType.FINAL: [
            PostProcessorType.NO_NEW_HEADERS,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE,
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.PRESERVE_F_STRING_TAGS,
            PostProcessorType.ENSURE_BLANK_LINE,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST,
        ],
        PhaseType.INTRODUCTION: [
            PostProcessorType.NO_NEW_HEADERS,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE,
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.PRESERVE_F_STRING_TAGS,
            PostProcessorType.ENSURE_BLANK_LINE,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST,
        ],
        PhaseType.SUMMARY: [
            PostProcessorType.REVERT_REMOVED_BLOCK_LINES,
            PostProcessorType.NO_NEW_HEADERS,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE,
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.PRESERVE_F_STRING_TAGS,
            PostProcessorType.ENSURE_BLANK_LINE,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST,
        ],
        PhaseType.ANNOTATE: [
            PostProcessorType.REVERT_REMOVED_BLOCK_LINES,
            PostProcessorType.ORDER_QUOTE_ANNOTATION,
            PostProcessorType.NO_NEW_HEADERS,
            PostProcessorType.REMOVE_TRAILING_WHITESPACE,
            PostProcessorType.REMOVE_XML_TAGS,
            PostProcessorType.PRESERVE_F_STRING_TAGS,
            PostProcessorType.ENSURE_BLANK_LINE,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST,
        ],
    }

    @staticmethod
    def create_standard_phase(
        config: PhaseConfig,
        length_reduction: Optional[Any] = None,
        tags_to_preserve: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
    ) -> StandardLlmPhase:
        """
        Create a standard LLM phase with optional post-processors.

        Args:
            config (PhaseConfig): Configuration object containing all phase parameters
            length_reduction: Length reduction parameter for the phase
            tags_to_preserve: Tags to preserve during processing
            max_workers: Maximum number of workers for parallel processing

        Returns:
            StandardLlmPhase: Configured standard phase
        """
        post_processor_chain = PhaseFactory._create_post_processor_chain(
            post_processors=config.post_processors, phase_type=config.phase_type, tags_to_preserve=tags_to_preserve
        )

        # Validate required fields
        if config.name is None:
            raise ValueError("name is required for StandardLlmPhase")
        if config.input_file_path is None:
            raise ValueError("input_file_path is required for StandardLlmPhase")
        if config.output_file_path is None:
            raise ValueError("output_file_path is required for StandardLlmPhase")
        if config.original_file_path is None:
            raise ValueError("original_file_path is required for StandardLlmPhase")
        if config.book_name is None:
            raise ValueError("book_name is required for StandardLlmPhase")
        if config.author_name is None:
            raise ValueError("author_name is required for StandardLlmPhase")

        return StandardLlmPhase(
            name=config.name,
            input_file_path=config.input_file_path,
            output_file_path=config.output_file_path,
            original_file_path=config.original_file_path,
            system_prompt_path=config.system_prompt_path,
            user_prompt_path=config.user_prompt_path,
            book_name=config.book_name,
            author_name=config.author_name,
            model=config.model,
            temperature=config.temperature,
            max_workers=max_workers,
            reasoning=config.reasoning,
            post_processor_chain=post_processor_chain,
            length_reduction=length_reduction,
        )

    @staticmethod
    def create_introduction_annotation_phase(
        config: PhaseConfig,
        length_reduction: Optional[Any] = None,
        tags_to_preserve: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
    ) -> IntroductionAnnotationPhase:
        """
        Create an introduction annotation phase with optional post-processors.

        Args:
            config (PhaseConfig): Configuration object containing all phase parameters
            length_reduction: Length reduction parameter for the phase
            tags_to_preserve: Tags to preserve during processing
            max_workers: Maximum number of workers for parallel processing

        Returns:
            IntroductionAnnotationPhase: Configured introduction annotation phase
        """
        post_processor_chain = PhaseFactory._create_post_processor_chain(
            post_processors=config.post_processors, phase_type=config.phase_type, tags_to_preserve=tags_to_preserve
        )

        # Validate required fields
        if config.name is None:
            raise ValueError("name is required for IntroductionAnnotationPhase")
        if config.input_file_path is None:
            raise ValueError("input_file_path is required for IntroductionAnnotationPhase")
        if config.output_file_path is None:
            raise ValueError("output_file_path is required for IntroductionAnnotationPhase")
        if config.original_file_path is None:
            raise ValueError("original_file_path is required for IntroductionAnnotationPhase")
        if config.book_name is None:
            raise ValueError("book_name is required for IntroductionAnnotationPhase")
        if config.author_name is None:
            raise ValueError("author_name is required for IntroductionAnnotationPhase")

        return IntroductionAnnotationPhase(
            name=config.name,
            input_file_path=config.input_file_path,
            output_file_path=config.output_file_path,
            original_file_path=config.original_file_path,
            system_prompt_path=config.system_prompt_path,
            user_prompt_path=config.user_prompt_path,
            book_name=config.book_name,
            author_name=config.author_name,
            model=config.model,
            temperature=config.temperature,
            max_workers=max_workers,
            reasoning=config.reasoning,
            post_processor_chain=post_processor_chain,
            length_reduction=length_reduction,
        )

    @staticmethod
    def create_summary_annotation_phase(
        config: PhaseConfig,
        length_reduction: Optional[Any] = None,
        tags_to_preserve: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
    ) -> SummaryAnnotationPhase:
        """
        Create a summary annotation phase with optional post-processors.

        Args:
            config (PhaseConfig): Configuration object containing all phase parameters
            length_reduction: Length reduction parameter for the phase
            tags_to_preserve: Tags to preserve during processing
            max_workers: Maximum number of workers for parallel processing

        Returns:
            SummaryAnnotationPhase: Configured summary annotation phase
        """
        post_processor_chain = PhaseFactory._create_post_processor_chain(
            post_processors=config.post_processors, phase_type=config.phase_type, tags_to_preserve=tags_to_preserve
        )

        # Validate required fields
        if config.name is None:
            raise ValueError("name is required for SummaryAnnotationPhase")
        if config.input_file_path is None:
            raise ValueError("input_file_path is required for SummaryAnnotationPhase")
        if config.output_file_path is None:
            raise ValueError("output_file_path is required for SummaryAnnotationPhase")
        if config.original_file_path is None:
            raise ValueError("original_file_path is required for SummaryAnnotationPhase")
        if config.book_name is None:
            raise ValueError("book_name is required for SummaryAnnotationPhase")
        if config.author_name is None:
            raise ValueError("author_name is required for SummaryAnnotationPhase")

        return SummaryAnnotationPhase(
            name=config.name,
            input_file_path=config.input_file_path,
            output_file_path=config.output_file_path,
            original_file_path=config.original_file_path,
            system_prompt_path=config.system_prompt_path,
            user_prompt_path=config.user_prompt_path,
            book_name=config.book_name,
            author_name=config.author_name,
            model=config.model,
            temperature=config.temperature,
            max_workers=max_workers,
            reasoning=config.reasoning,
            post_processor_chain=post_processor_chain,
            length_reduction=length_reduction,
        )

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
            "order_quote_annotation": OrderQuoteAnnotationProcessor,
            "no_new_headers": NoNewHeadersPostProcessor,
            "revert_removed_block_lines": RevertRemovedBlockLines,
            "remove_blank_lines_in_list": RemoveBlankLinesInListProcessor,
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
            PostProcessorType.ORDER_QUOTE_ANNOTATION: OrderQuoteAnnotationProcessor,
            PostProcessorType.NO_NEW_HEADERS: NoNewHeadersPostProcessor,
            PostProcessorType.REVERT_REMOVED_BLOCK_LINES: RevertRemovedBlockLines,
            PostProcessorType.PRESERVE_F_STRING_TAGS: PreserveFStringTagsProcessor,
            PostProcessorType.REMOVE_BLANK_LINES_IN_LIST: RemoveBlankLinesInListProcessor,
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
