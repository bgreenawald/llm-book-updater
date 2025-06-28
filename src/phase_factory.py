from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm_model import LlmModel
from src.llm_phase import IntroductionAnnotationPhase, StandardLlmPhase, SummaryAnnotationPhase
from src.post_processors import (
    ConsistencyPostProcessor,
    ContentValidationPostProcessor,
    CustomPostProcessor,
    EnsureBlankLineProcessor,
    RemoveXmlTagsProcessor,
    RemoveTrailingWhitespaceProcessor,
    OrderQuoteAnnotationProcessor,
    PostProcessorChain,
)


class PhaseFactory:
    """
    Factory class for creating different types of LLM phases with appropriate
    post-processor configurations.

    This allows for easy creation of phases with different post-processing needs
    without having to manually configure post-processor chains for each phase type.
    """

    @staticmethod
    def create_standard_phase(
        name: str,
        input_file_path: Path,
        output_file_path: Path,
        original_file_path: Path,
        system_prompt_path: Path,
        user_prompt_path: Path,
        book_name: str,
        author_name: str,
        model: LlmModel,
        temperature: float = 0.2,
        max_workers: Optional[int] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        post_processors: Optional[List[str]] = None,
        custom_post_processors: Optional[List[CustomPostProcessor]] = None,
    ) -> StandardLlmPhase:
        """
        Create a standard LLM phase with optional post-processors.

        Args:
            name (str): Name of the phase
            input_file_path (Path): Path to input file
            output_file_path (Path): Path to output file
            original_file_path (Path): Path to original file
            system_prompt_path (Path): Path to system prompt
            user_prompt_path (Path): Path to user prompt
            book_name (str): Name of the book
            author_name (str): Name of the author
            model (LlmModel): LLM model instance
            temperature (float): LLM temperature
            max_workers (Optional[int]): Maximum worker threads
            reasoning (Optional[Dict[str, Any]]): Reasoning configuration
            post_processors (Optional[List[str]]): List of built-in post-processor names
            custom_post_processors (Optional[List[CustomPostProcessor]]): Custom post-processors

        Returns:
            StandardLlmPhase: Configured standard phase
        """
        post_processor_chain = PhaseFactory._create_post_processor_chain(post_processors, custom_post_processors)

        return StandardLlmPhase(
            name=name,
            input_file_path=input_file_path,
            output_file_path=output_file_path,
            original_file_path=original_file_path,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            book_name=book_name,
            author_name=author_name,
            model=model,
            temperature=temperature,
            max_workers=max_workers,
            reasoning=reasoning,
            post_processor_chain=post_processor_chain,
        )

    @staticmethod
    def create_introduction_annotation_phase(
        name: str,
        input_file_path: Path,
        output_file_path: Path,
        original_file_path: Path,
        system_prompt_path: Path,
        user_prompt_path: Path,
        book_name: str,
        author_name: str,
        model: LlmModel,
        temperature: float = 0.2,
        max_workers: Optional[int] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        post_processors: Optional[List[str]] = None,
        custom_post_processors: Optional[List[CustomPostProcessor]] = None,
    ) -> IntroductionAnnotationPhase:
        """
        Create an introduction annotation phase with optional post-processors.

        Args:
            name (str): Name of the phase
            input_file_path (Path): Path to input file
            output_file_path (Path): Path to output file
            original_file_path (Path): Path to original file
            system_prompt_path (Path): Path to system prompt
            user_prompt_path (Path): Path to user prompt
            book_name (str): Name of the book
            author_name (str): Name of the author
            model (LlmModel): LLM model instance
            temperature (float): LLM temperature
            max_workers (Optional[int]): Maximum worker threads
            reasoning (Optional[Dict[str, Any]]): Reasoning configuration
            post_processors (Optional[List[str]]): List of built-in post-processor names
            custom_post_processors (Optional[List[CustomPostProcessor]]): Custom post-processors

        Returns:
            IntroductionAnnotationPhase: Configured introduction annotation phase
        """
        post_processor_chain = PhaseFactory._create_post_processor_chain(post_processors, custom_post_processors)

        return IntroductionAnnotationPhase(
            name=name,
            input_file_path=input_file_path,
            output_file_path=output_file_path,
            original_file_path=original_file_path,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            book_name=book_name,
            author_name=author_name,
            model=model,
            temperature=temperature,
            max_workers=max_workers,
            reasoning=reasoning,
            post_processor_chain=post_processor_chain,
        )

    @staticmethod
    def create_summary_annotation_phase(
        name: str,
        input_file_path: Path,
        output_file_path: Path,
        original_file_path: Path,
        system_prompt_path: Path,
        user_prompt_path: Path,
        book_name: str,
        author_name: str,
        model: LlmModel,
        temperature: float = 0.2,
        max_workers: Optional[int] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        post_processors: Optional[List[str]] = None,
        custom_post_processors: Optional[List[CustomPostProcessor]] = None,
    ) -> SummaryAnnotationPhase:
        """
        Create a summary annotation phase with optional post-processors.

        Args:
            name (str): Name of the phase
            input_file_path (Path): Path to input file
            output_file_path (Path): Path to output file
            original_file_path (Path): Path to original file
            system_prompt_path (Path): Path to system prompt
            user_prompt_path (Path): Path to user prompt
            book_name (str): Name of the book
            author_name (str): Name of the author
            model (LlmModel): LLM model instance
            temperature (float): LLM temperature
            max_workers (Optional[int]): Maximum worker threads
            reasoning (Optional[Dict[str, Any]]): Reasoning configuration
            post_processors (Optional[List[str]]): List of built-in post-processor names
            custom_post_processors (Optional[List[CustomPostProcessor]]): Custom post-processors

        Returns:
            SummaryAnnotationPhase: Configured summary annotation phase
        """
        post_processor_chain = PhaseFactory._create_post_processor_chain(post_processors, custom_post_processors)

        return SummaryAnnotationPhase(
            name=name,
            input_file_path=input_file_path,
            output_file_path=output_file_path,
            original_file_path=original_file_path,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            book_name=book_name,
            author_name=author_name,
            model=model,
            temperature=temperature,
            max_workers=max_workers,
            reasoning=reasoning,
            post_processor_chain=post_processor_chain,
        )

    @staticmethod
    def _create_post_processor_chain(
        post_processors: Optional[List[str]] = None,
        custom_post_processors: Optional[List[CustomPostProcessor]] = None,
    ) -> Optional[PostProcessorChain]:
        """
        Create a post-processor chain from built-in and custom post-processors.

        Args:
            post_processors (Optional[List[str]]): List of built-in post-processor names
            custom_post_processors (Optional[List[CustomPostProcessor]]): Custom post-processors

        Returns:
            Optional[PostProcessorChain]: Configured post-processor chain or None
        """
        if not post_processors and not custom_post_processors:
            return None

        chain = PostProcessorChain()

        # Add built-in post-processors
        if post_processors:
            for processor_name in post_processors:
                processor = PhaseFactory._create_built_in_processor(processor_name)
                if processor:
                    chain.add_processor(processor)

        # Add custom post-processors
        if custom_post_processors:
            for processor in custom_post_processors:
                chain.add_processor(processor)

        return chain

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
            "consistency": ConsistencyPostProcessor,
            "validation": ContentValidationPostProcessor,
            "ensure_blank_line": EnsureBlankLineProcessor,
            "remove_xml_tags": RemoveXmlTagsProcessor,
            "remove_trailing_whitespace": RemoveTrailingWhitespaceProcessor,
            "order_quote_annotation": OrderQuoteAnnotationProcessor,
        }

        processor_class = processors.get(processor_name.lower())
        if processor_class:
            return processor_class(name=processor_name)

        return None
