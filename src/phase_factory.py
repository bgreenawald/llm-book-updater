from typing import List, Optional, Union

from src.config import PhaseConfig
from src.llm_phase import IntroductionAnnotationPhase, StandardLlmPhase, SummaryAnnotationPhase
from src.post_processors import (
    EnsureBlankLineProcessor,
    NoNewHeadersPostProcessor,
    OrderQuoteAnnotationProcessor,
    PostProcessor,
    PostProcessorChain,
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

    @staticmethod
    def create_standard_phase(config: PhaseConfig) -> StandardLlmPhase:
        """
        Create a standard LLM phase with optional post-processors.

        Args:
            config (PhaseConfig): Configuration object containing all phase parameters

        Returns:
            StandardLlmPhase: Configured standard phase
        """
        post_processor_chain = PhaseFactory._create_post_processor_chain(config.post_processors)

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
            max_workers=config.max_workers,
            reasoning=config.reasoning,
            post_processor_chain=post_processor_chain,
        )

    @staticmethod
    def create_introduction_annotation_phase(config: PhaseConfig) -> IntroductionAnnotationPhase:
        """
        Create an introduction annotation phase with optional post-processors.

        Args:
            config (PhaseConfig): Configuration object containing all phase parameters

        Returns:
            IntroductionAnnotationPhase: Configured introduction annotation phase
        """
        post_processor_chain = PhaseFactory._create_post_processor_chain(config.post_processors)

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
            max_workers=config.max_workers,
            reasoning=config.reasoning,
            post_processor_chain=post_processor_chain,
        )

    @staticmethod
    def create_summary_annotation_phase(config: PhaseConfig) -> SummaryAnnotationPhase:
        """
        Create a summary annotation phase with optional post-processors.

        Args:
            config (PhaseConfig): Configuration object containing all phase parameters

        Returns:
            SummaryAnnotationPhase: Configured summary annotation phase
        """
        post_processor_chain = PhaseFactory._create_post_processor_chain(config.post_processors)

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
            max_workers=config.max_workers,
            reasoning=config.reasoning,
            post_processor_chain=post_processor_chain,
        )

    @staticmethod
    def _create_built_in_processor(processor_name: str):
        """
        Create a built-in post-processor by name.

        Args:
            processor_name (str): Name of the built-in processor

        Returns:
            PostProcessor: The created processor or None if not found
        """
        processors = {
            "ensure_blank_line": EnsureBlankLineProcessor,
            "remove_xml_tags": RemoveXmlTagsProcessor,
            "remove_trailing_whitespace": RemoveTrailingWhitespaceProcessor,
            "order_quote_annotation": OrderQuoteAnnotationProcessor,
            "no_new_headers": NoNewHeadersPostProcessor,
            "revert_removed_block_lines": RevertRemovedBlockLines,
        }

        processor_class = processors.get(processor_name.lower())
        if processor_class:
            return processor_class()

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
        chain.add_processor(RemoveXmlTagsProcessor())
        chain.add_processor(RemoveTrailingWhitespaceProcessor())
        chain.add_processor(EnsureBlankLineProcessor())
        chain.add_processor(OrderQuoteAnnotationProcessor())
        return chain

    @staticmethod
    def _create_post_processor_chain(
        post_processors: Optional[List[Union[str, PostProcessor]]] = None,
    ) -> Optional[PostProcessorChain]:
        """
        Create a post-processor chain from a unified list of post-processors.

        The list can contain:
        - Strings: Names of built-in post-processors (e.g., "ensure_blank_line")
        - PostProcessor instances: Custom post-processor objects
        - Special aliases: "formatting" for a predefined chain of formatting processors

        Args:
            post_processors (Optional[List[Union[str, PostProcessor]]]):
                Unified list of post-processors (strings or instances)

        Returns:
            Optional[PostProcessorChain]: Configured post-processor chain or None
        """
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
                        chain.add_processor(processor)
                else:
                    # Handle built-in processor by name
                    processor = PhaseFactory._create_built_in_processor(processor_item)
                    if processor:
                        chain.add_processor(processor)
            elif isinstance(processor_item, PostProcessor):
                # Handle custom PostProcessor instances
                chain.add_processor(processor_item)
            else:
                # Skip invalid items
                continue

        return chain
