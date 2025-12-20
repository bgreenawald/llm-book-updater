import json
import re
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import tiktoken
from loguru import logger
from tqdm import tqdm

from src.constants import (
    DEFAULT_GENERATION_MAX_RETRIES,
    DEFAULT_MAX_SUBBLOCK_TOKENS,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_SUBBLOCK_TOKENS,
    DEFAULT_TAGS_TO_PRESERVE,
    LLM_DEFAULT_TEMPERATURE,
)
from src.cost_tracking_wrapper import add_generation_id
from src.llm_model import (
    GenerationFailedError,
    LlmModel,
    MaxRetriesExceededError,
    ResponseTruncatedError,
    is_failed_response,
)
from src.post_processors import EmptySectionError, PostProcessorChain


class LlmPhase(ABC):
    """
    Abstract base class for LLM processing phases.

    This class provides the core functionality for processing markdown content
    through LLM models, including file I/O, block processing, and post-processing.
    """

    def __init__(
        self,
        name: str,
        input_file_path: Path,
        output_file_path: Path,
        original_file_path: Path,
        system_prompt_path: Optional[Path],
        user_prompt_path: Optional[Path],
        book_name: str,
        author_name: str,
        model: LlmModel,
        temperature: float = LLM_DEFAULT_TEMPERATURE,
        max_workers: Optional[int] = None,
        reasoning: Optional[Dict[str, str]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        post_processor_chain: Optional[PostProcessorChain] = None,
        use_batch: bool = False,
        batch_size: Optional[int] = None,
        enable_retry: bool = False,
        max_retries: int = DEFAULT_GENERATION_MAX_RETRIES,
        use_subblocks: bool = False,
        max_subblock_tokens: int = DEFAULT_MAX_SUBBLOCK_TOKENS,
        min_subblock_tokens: int = DEFAULT_MIN_SUBBLOCK_TOKENS,
        skip_if_less_than_tokens: Optional[int] = None,
    ) -> None:
        """
        Initialize the LLM phase with all necessary parameters.

        Args:
            name (str): Name of the phase for logging and identification
            input_file_path (Path): Path to the input markdown file
            output_file_path (Path): Path where the processed output will be written
            original_file_path (Path): Path to the original markdown file for reference
            system_prompt_path (Optional[Path]): Path to the system prompt file
            user_prompt_path (Optional[Path]): Path to the user prompt file
            book_name (str): Name of the book being processed
            author_name (str): Name of the book's author
            model (LlmModel): LLM model instance for making API calls
            temperature (float): Temperature setting for the LLM model
            max_workers (Optional[int]): Maximum number of worker threads for parallel processing
            reasoning (Optional[Dict[str, str]]): Reasoning configuration for the model
            llm_kwargs (Optional[Dict[str, Any]]): Additional kwargs to pass to LLM calls
                (e.g., provider for OpenRouter)
            post_processor_chain (Optional[PostProcessorChain]): Chain of post-processors to apply
            use_batch (bool): Whether to use batch processing for LLM calls (if supported)
            batch_size (Optional[int]): Number of items to process in each batch (if None, processes all blocks at once)
            enable_retry (bool): Whether to retry failed generations. When False (default),
                any generation failure immediately stops the pipeline.
            max_retries (int): Maximum number of retry attempts per generation (default: 2)
            use_subblocks (bool): Whether to split large blocks into smaller sub-blocks for processing
            max_subblock_tokens (int): Maximum tokens per sub-block when use_subblocks is enabled
            min_subblock_tokens (int): Minimum tokens per sub-block when grouping paragraphs
            skip_if_less_than_tokens (Optional[int]): If set, blocks with fewer tokens than this value
                will be skipped entirely (before chunking into subblocks). Default None means no skipping.
        """
        self.name = name
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.original_file_path = original_file_path
        self.system_prompt_path = system_prompt_path
        self.user_prompt_path = user_prompt_path
        self.book_name = book_name
        self.author_name = author_name
        self.model = model
        self.temperature = temperature
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        self.reasoning = reasoning or {}
        self.llm_kwargs = llm_kwargs or {}
        self.post_processor_chain = post_processor_chain
        self.use_batch = use_batch
        self.batch_size = batch_size
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.use_subblocks = use_subblocks
        self.max_subblock_tokens = max_subblock_tokens
        self.min_subblock_tokens = min_subblock_tokens
        self.skip_if_less_than_tokens = skip_if_less_than_tokens

        # Sub-block processing stats (only used when self.use_subblocks is True)
        self._subblock_stats_lock = threading.Lock()
        self._subblocks_processed_total: int = 0
        self._subblock_blocks_processed_total: int = 0
        self._max_subblocks_in_single_block: int = 0

        # Initialize content storage
        self.input_text = ""
        self.original_text = ""
        self.system_prompt = ""
        self.user_prompt = ""
        self.content = ""

        # Initialize token counting
        self.start_token_count: Optional[int] = None
        self.end_token_count: Optional[int] = None
        self._tokenizer = tiktoken.get_encoding("cl100k_base")  # Use GPT-4/GPT-3.5 encoding

        # Initialize logging
        logger.info(f"Initializing LlmPhase: {name}")
        logger.debug(f"Input file: {input_file_path}")
        logger.debug(f"Original file: {original_file_path}")
        logger.debug(f"Output file: {output_file_path}")
        logger.debug(f"System prompt path: {system_prompt_path}")
        logger.debug(f"User prompt path: {user_prompt_path}")
        logger.debug(f"Book: {book_name} by {author_name}")
        logger.debug(f"Temperature: {temperature}, Max workers: {max_workers}")
        if use_subblocks:
            logger.debug(f"Sub-block processing enabled: min={min_subblock_tokens}, max={max_subblock_tokens} tokens")
            logger.info(
                "Sub-block processing enabled for phase "
                f"'{self.name}': min_subblock_tokens={min_subblock_tokens}, max_subblock_tokens={max_subblock_tokens}"
            )

        try:
            # Load all necessary files
            logger.info("Reading input file and system prompt")
            logger.debug("Reading input file")
            self.input_text = self._read_input_file()
            logger.debug(f"Read {len(self.input_text)} characters from input file")
            logger.debug("Reading original file")
            self.original_text = self._read_original_file()
            logger.debug(f"Read {len(self.original_text)} characters from original file")
            logger.debug("Reading system prompt")
            self.system_prompt = self._read_system_prompt()
            logger.debug("System prompt loaded successfully")
            logger.debug("Reading user prompt")
            self.user_prompt = self._read_user_prompt()
            logger.debug("User prompt loaded successfully")
            logger.debug(f"Reasoning configuration: {self.reasoning}")
        except Exception as e:
            logger.error(f"Failed to initialize LlmPhase: {str(e)}")
            raise

    def __str__(self) -> str:
        """
        Returns a string representation of the LlmPhase instance.

        Returns:
            str: String representation of the phase
        """
        return (
            f"LlmPhase(name={self.name}, "
            f"input_file={self.input_file_path}, "
            f"output_file={self.output_file_path}, "
            f"book={self.book_name}, "
            f"author={self.author_name}, "
            f"temperature={self.temperature}, "
            f"max_workers={self.max_workers})"
        )

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the LlmPhase instance for debugging.

        Returns:
            str: Detailed string representation of the phase
        """
        return self.__str__()

    def _count_tokens(self, text: str) -> int:
        """
        Count the approximate number of tokens in a text using tiktoken.

        Args:
            text (str): The text to count tokens for

        Returns:
            int: The approximate number of tokens
        """
        try:
            return len(self._tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}, returning character count / 4 as fallback")
            # Fallback: rough approximation of 1 token per 4 characters
            return len(text) // 4

    def _should_skip_block_by_tokens(self, current_block: str) -> bool:
        """
        Check if a block should be skipped based on token count threshold.

        Args:
            current_block (str): The block to check

        Returns:
            bool: True if the block should be skipped, False otherwise
        """
        if self.skip_if_less_than_tokens is None:
            return False

        token_count = self._count_tokens(current_block)
        should_skip = token_count < self.skip_if_less_than_tokens

        if should_skip:
            logger.debug(f"Skipping block with {token_count} tokens (threshold: {self.skip_if_less_than_tokens})")

        return should_skip

    @abstractmethod
    def _process_block(self, current_block: str, original_block: str, **kwargs) -> str:
        """
        Process a single markdown block using the LLM model.

        This is an abstract method that must be implemented by subclasses.
        It defines how individual markdown blocks are processed by the LLM.

        Args:
            current_block (str): The current markdown block to process
            original_block (str): The original markdown block for reference
            **kwargs: Additional context or parameters

        Returns:
            str: The processed block content
        """
        pass

    def _apply_post_processing(self, original_block: str, llm_block: str, **kwargs) -> str:
        """
        Apply post-processing chain to the LLM-generated block.

        This method applies a series of post-processors to clean up and improve
        the LLM-generated content, using the original block as a reference.

        Args:
            original_block (str): The original markdown block before LLM processing
            llm_block (str): The block generated by the LLM
            **kwargs: Additional context or parameters

        Returns:
            str: The post-processed block
        """
        if not self.post_processor_chain:
            return llm_block

        try:
            logger.debug(f"Applying post-processing chain with {len(self.post_processor_chain)} processors")
            logger.debug("Starting post-processing")
            processed_block = self.post_processor_chain.process(
                original_block=original_block, llm_block=llm_block, **kwargs
            )
            logger.debug("Post-processing completed successfully")
            return processed_block
        except EmptySectionError:
            # EmptySectionError is a critical validation error that should stop the pipeline
            logger.error("EmptySectionError in post-processing - propagating to stop pipeline")
            raise
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            # Common post-processor errors are logged but non-fatal; return unprocessed block
            logger.error(f"Error during post-processing: {str(e)}")
            logger.exception("Post-processing error stack trace")
            return llm_block
        except Exception as e:
            # Catch-all for unexpected errors; log and return unprocessed block
            logger.error(f"Unexpected error during post-processing: {str(e)}")
            logger.exception("Unexpected post-processing error stack trace")
            return llm_block

    def _make_llm_call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        block_info: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Make an LLM call with retry logic based on phase configuration.

        If enable_retry is False (default), any failure immediately raises
        GenerationFailedError to stop the pipeline.

        If enable_retry is True, failures are retried up to max_retries times
        before raising MaxRetriesExceededError.

        Args:
            system_prompt: The system prompt for the LLM call
            user_prompt: The user prompt for the LLM call
            block_info: Optional identifier for the block being processed (for error messages)
            **kwargs: Additional arguments to pass to the LLM call

        Returns:
            Tuple[str, str]: (response_content, generation_id)

        Raises:
            GenerationFailedError: If retry is disabled and the call fails
            MaxRetriesExceededError: If all retry attempts are exhausted
        """
        last_error: Optional[Exception] = None
        last_content: str = ""

        # Determine number of attempts: 1 if no retry, else max_retries + 1 (initial + retries)
        max_attempts = (self.max_retries + 1) if self.enable_retry else 1

        for attempt in range(max_attempts):
            try:
                # Make the LLM call
                content, generation_id = self.model.chat_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    **kwargs,
                )

                # Check if the response indicates a failure
                if is_failed_response(content):
                    last_content = content
                    error_msg = f"LLM returned failed response: {content[:100]}..."
                    if self.enable_retry and attempt < self.max_retries:
                        logger.warning(
                            f"Generation failed (attempt {attempt + 1}/{max_attempts}): {error_msg}. Retrying..."
                        )
                        continue
                    else:
                        # No retry or retries exhausted
                        if self.enable_retry:
                            raise MaxRetriesExceededError(
                                message=f"Generation failed after {max_attempts} attempts: {error_msg}",
                                attempts=max_attempts,
                                block_info=block_info,
                            )
                        else:
                            raise GenerationFailedError(
                                message=error_msg,
                                block_info=block_info,
                            )

                # Success!
                return content, generation_id

            except (GenerationFailedError, MaxRetriesExceededError):
                # Re-raise our custom exceptions (these are terminal failures)
                raise
            except ResponseTruncatedError as e:
                # Truncation is retryable - model may use fewer reasoning tokens on retry
                last_error = e
                error_msg = f"Response truncated: {str(e)}"
                if self.enable_retry and attempt < self.max_retries:
                    logger.warning(
                        f"Response truncated (attempt {attempt + 1}/{max_attempts}): {error_msg}. Retrying..."
                    )
                    continue
                else:
                    # No retry or retries exhausted
                    if self.enable_retry:
                        raise MaxRetriesExceededError(
                            message=f"Response truncated after {max_attempts} attempts: {error_msg}",
                            attempts=max_attempts,
                            block_info=block_info,
                        ) from e
                    else:
                        raise GenerationFailedError(
                            message=error_msg,
                            block_info=block_info,
                        ) from e
            except Exception as e:
                last_error = e
                error_msg = f"LLM call failed: {str(e)}"
                if self.enable_retry and attempt < self.max_retries:
                    logger.warning(
                        f"Generation failed (attempt {attempt + 1}/{max_attempts}): {error_msg}. Retrying..."
                    )
                    continue
                else:
                    # No retry or retries exhausted
                    if self.enable_retry:
                        raise MaxRetriesExceededError(
                            message=f"Generation failed after {max_attempts} attempts: {error_msg}",
                            attempts=max_attempts,
                            block_info=block_info,
                        ) from e
                    else:
                        raise GenerationFailedError(
                            message=error_msg,
                            block_info=block_info,
                        ) from e

        # This should never be reached, but just in case
        if last_error:
            raise GenerationFailedError(
                message=f"LLM call failed: {str(last_error)}",
                block_info=block_info,
            ) from last_error
        else:
            raise GenerationFailedError(
                message=f"LLM returned failed response: {last_content[:100]}...",
                block_info=block_info,
            )

    def _read_input_file(self) -> str:
        """
        Read the input markdown file.

        This method reads the input file that contains the markdown content
        to be processed by the LLM phase.

        Returns:
            str: The content of the input file

        Raises:
            FileNotFoundError: If the input file does not exist
            Exception: If there's an error reading the file
        """
        try:
            if not self.input_file_path.exists():
                logger.error(f"Input file not found: {self.input_file_path}")
                raise FileNotFoundError(f"Input file not found: {self.input_file_path}")

            with self.input_file_path.open(mode="r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Successfully read input file: {self.input_file_path}")
            return content
        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading input file {self.input_file_path}: {str(e)}")
            raise

    def _read_original_file(self) -> str:
        """
        Read the original markdown file.

        This method reads the original file that serves as a reference
        for the markdown content being processed.

        Returns:
            str: The content of the original file

        Raises:
            FileNotFoundError: If the original file does not exist
            Exception: If there's an error reading the file
        """
        try:
            if not self.original_file_path.exists():
                logger.error(f"Original file not found: {self.original_file_path}")
                raise FileNotFoundError(f"Original file not found: {self.original_file_path}")

            with self.original_file_path.open(mode="r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Successfully read original file: {self.original_file_path}")
            return content
        except FileNotFoundError:
            logger.error(f"Original file not found: {self.original_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading original file {self.original_file_path}: {str(e)}")
            raise

    def _write_output_file(self, content: str) -> None:
        """
        Write the processed content to the output file.

        This method writes the final processed content to the output file,
        creating the output directory if it doesn't exist.

        Args:
            content (str): The content to write to the output file

        Raises:
            Exception: If there's an error writing the file
        """
        try:
            # Ensure output directory exists
            self.output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_file_path.open(mode="w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Successfully wrote output to: {self.output_file_path}")
            logger.debug(f"Wrote {len(content)} characters to output file")
        except Exception as e:
            logger.error(f"Error writing to output file {self.output_file_path}: {str(e)}")
            raise

    def _read_system_prompt(self) -> str:
        """
        Read and format the system prompt file.

        This method reads the system prompt file and formats it with any
        necessary parameters, such as length reduction settings.

        Returns:
            str: The formatted system prompt content

        Raises:
            FileNotFoundError: If the system prompt file does not exist
            Exception: If there's an error reading or formatting the file
        """
        try:
            if not self.system_prompt_path or not self.system_prompt_path.exists():
                logger.error(f"System prompt file not found: {self.system_prompt_path}")
                raise FileNotFoundError(f"System prompt file not found: {self.system_prompt_path}")

            with self.system_prompt_path.open(mode="r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Read system prompt from {self.system_prompt_path}")

            # Format the system prompt with parameters if needed
            format_params = {}

            # Get tags to preserve from post-processor chain if available
            tags_to_preserve = DEFAULT_TAGS_TO_PRESERVE
            if self.post_processor_chain:
                for processor in self.post_processor_chain.processors:
                    if hasattr(processor, "tags_to_preserve"):
                        tags_to_preserve = processor.tags_to_preserve
                        break

            # Add all tags_to_preserve as format parameters with their original values
            for tag in tags_to_preserve:
                # Extract the tag name without braces
                tag_name = tag.strip("{}")
                format_params[tag_name] = tag

            if format_params:
                try:
                    content = content.format(**format_params)
                    logger.debug(f"Formatted system prompt with parameters: {format_params}")
                except KeyError as e:
                    logger.warning(f"System prompt contains undefined parameter: {e}")
                    raise
                except Exception as e:
                    logger.warning(f"Error formatting system prompt: {e}")
                    # Return unformatted content on any other formatting error
                    logger.warning("Returning unformatted system prompt due to formatting errors")
                    raise

            return content
        except FileNotFoundError:
            logger.error(f"System prompt file not found: {self.system_prompt_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading system prompt file {self.system_prompt_path}: {str(e)}")
            raise

    def _read_user_prompt(self) -> str:
        """
        Read the user prompt file.

        This method reads the user prompt file that will be used to
        format user messages for the LLM.

        Returns:
            str: The content of the user prompt file

        Raises:
            FileNotFoundError: If the user prompt file does not exist
            Exception: If there's an error reading the file
        """
        try:
            if not self.user_prompt_path or not self.user_prompt_path.exists():
                logger.error(f"User prompt file not found: {self.user_prompt_path}")
                raise FileNotFoundError(f"User prompt file not found: {self.user_prompt_path}")

            with self.user_prompt_path.open(mode="r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Read user prompt from {self.user_prompt_path}")
            return content
        except FileNotFoundError:
            logger.error(f"User prompt file not found: {self.user_prompt_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading user prompt file {self.user_prompt_path}: {str(e)}")
            raise

    def _format_user_message(
        self, current_body: str, original_body: str, current_header: str, original_header: str
    ) -> str:
        """
        Format the user message for the LLM using the user prompt template.

        This method formats the user message by substituting placeholders
        in the user prompt template with actual content from the markdown blocks.

        Args:
            current_body (str): The body of the current markdown block
            original_body (str): The body of the original markdown block
            current_header (str): The header of the current markdown block
            original_header (str): The header of the original markdown block

        Returns:
            str: The formatted user message
        """
        context = {
            "current_body": current_body,
            "original_body": original_body,
            "current_header": current_header,
            "original_header": original_header,
            "book_name": self.book_name,
            "author_name": self.author_name,
        }

        try:
            ret = self.user_prompt.format(**context)
            return ret
        except KeyError as e:
            logger.warning(f"User prompt contains undefined parameter: {e}")
            return self.user_prompt

    def _get_header_and_body(self, block: str) -> Tuple[str, str]:
        """
        Split a markdown block into header and body components.

        This method separates the first line (header) from the rest of the
        content (body) in a markdown block.

        Args:
            block (str): The markdown block to split

        Returns:
            Tuple[str, str]: A tuple containing (header, body)
        """
        lines = block.strip().split("\n", 1)
        header = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        return header, body

    def _contains_only_special_tags(self, body: str) -> bool:
        """
        Check if a body contains only special tags (like {preface}, {license}) after removing blank lines.

        Args:
            body (str): The body content to check

        Returns:
            bool: True if the body contains only special tags, False otherwise
        """
        if not body.strip():
            return True

        # Get the tags to preserve from the post-processor chain if available
        tags_to_preserve = ["{preface}", "{license}"]  # Default tags
        if self.post_processor_chain:
            for processor in self.post_processor_chain.processors:
                if hasattr(processor, "tags_to_preserve"):
                    tags_to_preserve = processor.tags_to_preserve
                    break

        # Split into lines and remove blank lines
        lines = [line.strip() for line in body.split("\n") if line.strip()]

        # Check if all non-blank lines are special tags
        for line in lines:
            if line not in tags_to_preserve:
                return False

        return len(lines) > 0  # Must have at least one tag to be considered "only tags"

    def _split_body_into_paragraphs(self, body: str) -> List[str]:
        """
        Split a body text into paragraphs on single newlines.

        Each line in the content represents a separate paragraph.
        Empty lines are filtered out.

        Args:
            body: The body text to split (header should already be removed)

        Returns:
            List of paragraph strings, with empty lines filtered out
        """
        if not body.strip():
            return []

        # Split on single newlines - each line is a paragraph
        lines = body.split("\n")

        # Filter out empty lines and strip whitespace
        paragraphs = [line.strip() for line in lines if line.strip()]

        return paragraphs

    def _group_paragraphs_into_subblocks(self, paragraphs: List[str]) -> List[str]:
        """
        Group paragraphs into sub-blocks based on token bounds.

        Groups consecutive paragraphs until min_subblock_tokens is reached.
        Large paragraphs exceeding max_subblock_tokens are kept as lone sub-blocks
        (never split a paragraph). If the final group is below min_subblock_tokens,
        it is combined with the second-to-last group and redistributed evenly.

        Args:
            paragraphs: List of paragraph strings to group

        Returns:
            List of sub-block strings (concatenated paragraphs with single newline separators)
        """
        if not paragraphs:
            return []

        if len(paragraphs) == 1:
            return [paragraphs[0]]

        # Calculate token count for each paragraph
        paragraph_tokens = [self._count_tokens(p) for p in paragraphs]

        # Group paragraphs into sub-blocks
        groups: List[List[int]] = []  # List of paragraph indices for each group
        current_group: List[int] = []
        current_tokens = 0

        for i, (para, tokens) in enumerate(zip(paragraphs, paragraph_tokens)):
            # If single paragraph exceeds max, keep it as its own group
            if tokens >= self.max_subblock_tokens:
                # Finish current group first if it has content
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_tokens = 0
                # Add large paragraph as its own group
                groups.append([i])
                continue

            # Add to current group
            current_group.append(i)
            current_tokens += tokens

            # If we've reached min_subblock_tokens, start a new group
            if current_tokens >= self.min_subblock_tokens:
                groups.append(current_group)
                current_group = []
                current_tokens = 0

        # Handle any remaining paragraphs
        if current_group:
            groups.append(current_group)

        # Handle small trailing chunk: redistribute with second-to-last group
        if len(groups) >= 2:
            last_group = groups[-1]
            last_group_tokens = sum(paragraph_tokens[i] for i in last_group)

            if last_group_tokens < self.min_subblock_tokens:
                # Combine last two groups and redistribute
                second_last_group = groups[-2]

                # Check if second-to-last group is a single large paragraph
                # In that case, we can't redistribute
                second_last_tokens = sum(paragraph_tokens[i] for i in second_last_group)
                if len(second_last_group) == 1 and second_last_tokens >= self.max_subblock_tokens:
                    # Can't redistribute with oversized paragraph, leave as is
                    pass
                else:
                    # Merge and redistribute
                    combined_indices = second_last_group + last_group
                    combined_tokens = second_last_tokens + last_group_tokens

                    # Find optimal split point for even distribution
                    target_tokens = combined_tokens // 2
                    running_tokens = 0
                    split_point = 0

                    for j, idx in enumerate(combined_indices):
                        running_tokens += paragraph_tokens[idx]
                        if running_tokens >= target_tokens:
                            # Check which is closer to target: before or after this paragraph
                            before = running_tokens - paragraph_tokens[idx]
                            after = running_tokens
                            if abs(target_tokens - before) < abs(target_tokens - after) and j > 0:
                                split_point = j
                            else:
                                split_point = j + 1
                            break
                    else:
                        split_point = len(combined_indices)

                    # Ensure we have at least one paragraph in each group
                    if split_point == 0:
                        split_point = 1
                    elif split_point >= len(combined_indices):
                        split_point = len(combined_indices) - 1

                    # Replace last two groups with redistributed groups
                    groups = groups[:-2]
                    groups.append(combined_indices[:split_point])
                    groups.append(combined_indices[split_point:])

        # Convert groups to sub-block strings
        subblocks = []
        for group in groups:
            subblock_paragraphs = [paragraphs[i] for i in group]
            subblocks.append("\n".join(subblock_paragraphs))

        return subblocks

    def _assemble_processed_block(
        self, current_header: str, current_body: str, llm_response: str, original_body: str, **kwargs
    ) -> str:
        """
        Assemble the final processed block from components.

        This method can be overridden by subclasses to customize how the block is assembled.
        For example, IntroductionAnnotationPhase prepends the LLM response, while
        SummaryAnnotationPhase appends it.

        Args:
            current_header: The header of the current block
            current_body: The body of the current block
            llm_response: The processed response from the LLM
            original_body: The original body for reference
            **kwargs: Additional context

        Returns:
            The assembled block as a string
        """
        # Default behavior: replace the body with the LLM response
        if llm_response.strip():
            return f"{current_header}\n\n{llm_response}\n\n"
        else:
            return f"{current_header}\n\n"

    def _process_batch(self, batch: List[Tuple[str, str]], **kwargs) -> List[str]:
        """
        Process a batch of markdown blocks using batch API if available.

        Args:
            batch: List of tuples containing (current_block, original_block)
            **kwargs: Additional arguments to pass to the processing methods

        Returns:
            List of processed block strings
        """
        try:
            # Check if the model supports batch processing
            if hasattr(self.model, "supports_batch") and self.model.supports_batch():
                logger.debug(f"Processing batch of {len(batch)} blocks using batch API")

                # Sub-block aware batching: when enabled, each block can expand to multiple batch requests.
                if self.use_subblocks:
                    block_entries: List[Dict[str, Any]] = []
                    valid_requests: List[Dict[str, Any]] = []

                    for block_index, (current_block, original_block) in enumerate(batch):
                        current_header, current_body = self._get_header_and_body(block=current_block)
                        original_header, original_body = self._get_header_and_body(block=original_block)

                        # Skip blocks based on token count threshold (before subblock processing)
                        if self._should_skip_block_by_tokens(current_block):
                            block_entries.append(
                                {
                                    "skipped": True,
                                    "current_header": current_header,
                                    "current_body": current_body,
                                }
                            )
                            continue

                        # Skip empty blocks or blocks with only special tags
                        if (not current_body.strip() and not original_body.strip()) or (
                            self._contains_only_special_tags(current_body)
                            and self._contains_only_special_tags(original_body)
                        ):
                            block_entries.append(
                                {
                                    "skipped": True,
                                    "current_header": current_header,
                                    "current_body": current_body,
                                }
                            )
                            continue

                        current_paragraphs = self._split_body_into_paragraphs(current_body)
                        original_paragraphs = self._split_body_into_paragraphs(original_body)

                        # If no paragraphs, treat as a single request (preserves previous batch behavior).
                        if not current_paragraphs:
                            user_message = self._format_user_message(
                                current_body=current_body,
                                original_body=original_body,
                                current_header=current_header,
                                original_header=original_header,
                            )
                            req = {
                                "system_prompt": self.system_prompt,
                                "user_prompt": user_message,
                                "metadata": {
                                    "block_index": block_index,
                                    "subblock_index": 0,
                                    "subblock_count": 1,
                                    "current_header": current_header,
                                    "current_body_full": current_body,
                                    "original_body_full": original_body,
                                    "original_header": original_header,
                                },
                            }
                            block_entries.append(
                                {
                                    "skipped": False,
                                    "block_index": block_index,
                                    "current_header": current_header,
                                    "current_body_full": current_body,
                                    "original_body_full": original_body,
                                    "original_header": original_header,
                                    "subblock_count": 1,
                                }
                            )
                            valid_requests.append(req)
                            continue

                        current_subblocks = self._group_paragraphs_into_subblocks(current_paragraphs)
                        original_subblocks = self._group_paragraphs_into_subblocks(original_paragraphs)

                        # Ensure same number of sub-blocks (mirror non-batch subblock path behavior)
                        if len(current_subblocks) != len(original_subblocks):
                            logger.warning(
                                f"Sub-block count mismatch: current={len(current_subblocks)}, "
                                f"original={len(original_subblocks)}. Using current body grouping for original as well."
                            )
                            original_subblocks = self._group_paragraphs_into_subblocks(original_paragraphs)
                            if len(current_subblocks) != len(original_subblocks):
                                while len(original_subblocks) < len(current_subblocks):
                                    original_subblocks.append("")
                                original_subblocks = original_subblocks[: len(current_subblocks)]

                        with self._subblock_stats_lock:
                            self._subblock_blocks_processed_total += 1
                            self._subblocks_processed_total += len(current_subblocks)
                            if len(current_subblocks) > self._max_subblocks_in_single_block:
                                self._max_subblocks_in_single_block = len(current_subblocks)

                        logger.debug(
                            f"Batch mode: split block '{current_header[:30]}...' "
                            f"into {len(current_subblocks)} sub-blocks"
                        )

                        block_entries.append(
                            {
                                "skipped": False,
                                "block_index": block_index,
                                "current_header": current_header,
                                "current_body_full": current_body,
                                "original_body_full": original_body,
                                "original_header": original_header,
                                "subblock_count": len(current_subblocks),
                            }
                        )

                        for subblock_index, (curr_sb, orig_sb) in enumerate(zip(current_subblocks, original_subblocks)):
                            user_message = self._format_user_message(
                                current_body=curr_sb,
                                original_body=orig_sb,
                                current_header=current_header,
                                original_header=original_header,
                            )
                            valid_requests.append(
                                {
                                    "system_prompt": self.system_prompt,
                                    "user_prompt": user_message,
                                    "metadata": {
                                        "block_index": block_index,
                                        "subblock_index": subblock_index,
                                        "subblock_count": len(current_subblocks),
                                        "current_header": current_header,
                                        "current_body_full": current_body,
                                        "original_body_full": original_body,
                                        "original_header": original_header,
                                    },
                                }
                            )

                    if valid_requests:
                        call_kwargs = {**self.llm_kwargs, "reasoning": self.reasoning, **kwargs}
                        batch_responses = self.model.batch_chat_completion(
                            requests=valid_requests, temperature=self.temperature, **call_kwargs
                        )
                    else:
                        batch_responses = []

                    failed_response_indices: List[int] = []
                    for idx, response in enumerate(batch_responses):
                        if response.get("failed", False):
                            failed_response_indices.append(idx)
                    if failed_response_indices:
                        logger.warning(
                            f"Batch processing had {len(failed_response_indices)} failed responses out of "
                            f"{len(batch_responses)}"
                        )

                        if not self.enable_retry:
                            first_failed = batch_responses[failed_response_indices[0]]
                            failed_content = first_failed.get("content", "Unknown error")
                            block_info = f"batch index {failed_response_indices[0]}"
                            raise GenerationFailedError(
                                message=f"Batch generation failed (retry disabled): {failed_content[:200]}",
                                block_info=block_info,
                            )

                        logger.info(f"Retrying {len(failed_response_indices)} failed batch responses individually")
                        for failed_idx in failed_response_indices:
                            original_request = valid_requests[failed_idx]
                            header = original_request["metadata"].get("current_header", "unknown")[:50]
                            block_info = f"batch index {failed_idx}, header: {header}"

                            call_kwargs_retry = {**self.llm_kwargs, "reasoning": self.reasoning, **kwargs}
                            retried_content, retried_gen_id = self._make_llm_call_with_retry(
                                system_prompt=original_request["system_prompt"],
                                user_prompt=original_request["user_prompt"],
                                block_info=block_info,
                                **call_kwargs_retry,
                            )

                            batch_responses[failed_idx] = {
                                "content": retried_content,
                                "generation_id": retried_gen_id,
                                "metadata": original_request["metadata"],
                                "failed": False,
                            }

                    # Group responses back into blocks
                    subblock_outputs: Dict[int, List[Optional[str]]] = {}
                    for req, resp in zip(valid_requests, batch_responses):
                        req_dict: Dict[str, Any] = cast(Dict[str, Any], req)
                        subblock_metadata: Dict[str, Any] = cast(Dict[str, Any], req_dict.get("metadata", {}))
                        block_index = int(subblock_metadata["block_index"])
                        subblock_index = int(subblock_metadata["subblock_index"])
                        subblock_count = int(subblock_metadata["subblock_count"])
                        if block_index not in subblock_outputs:
                            subblock_outputs[block_index] = [None] * subblock_count
                        subblock_outputs[block_index][subblock_index] = resp.get("content", "")

                        generation_id = resp.get("generation_id")
                        if generation_id:
                            add_generation_id(phase_name=self.name, generation_id=generation_id)

                    # Reconstruct processed blocks maintaining original order
                    processed_blocks: List[str] = []
                    for entry in block_entries:
                        if entry.get("skipped", False):
                            processed_blocks.append(f"{entry['current_header']}\n\n{entry['current_body']}\n\n")
                            continue

                        block_index = int(entry["block_index"])
                        parts = subblock_outputs.get(block_index, [])
                        # Defensive: fill any missing subblocks with empty strings.
                        processed_body_raw = "\n".join([(p if p is not None else "") for p in parts])

                        processed_body = self._apply_post_processing(
                            original_block=entry["current_body_full"], llm_block=processed_body_raw, **kwargs
                        )
                        processed_block = self._assemble_processed_block(
                            current_header=entry["current_header"],
                            current_body=entry["current_body_full"],
                            llm_response=processed_body,
                            original_body=entry["original_body_full"],
                            **kwargs,
                        )
                        processed_blocks.append(processed_block)

                    return processed_blocks

                # Prepare batch requests (non-subblock batching: one request per block)
                batch_requests: List[Optional[Dict[str, Any]]] = []
                for current_block, original_block in batch:
                    current_header, current_body = self._get_header_and_body(block=current_block)
                    original_header, original_body = self._get_header_and_body(block=original_block)

                    # Skip blocks based on token count threshold (before processing)
                    if self._should_skip_block_by_tokens(current_block):
                        batch_requests.append(None)  # Mark as skip
                        continue

                    # Skip empty blocks or blocks with only special tags
                    if (not current_body.strip() and not original_body.strip()) or (
                        self._contains_only_special_tags(current_body)
                        and self._contains_only_special_tags(original_body)
                    ):
                        batch_requests.append(None)  # Mark as skip
                        continue

                    # Format the user message
                    user_message = self._format_user_message(
                        current_body=current_body,
                        original_body=original_body,
                        current_header=current_header,
                        original_header=original_header,
                    )

                    batch_requests.append(
                        {
                            "system_prompt": self.system_prompt,
                            "user_prompt": user_message,
                            "metadata": {
                                "current_header": current_header,
                                "current_body": current_body,
                                "original_body": original_body,
                                "original_header": original_header,
                            },
                        }
                    )

                # Process non-None requests through batch API
                valid_requests = [req for req in batch_requests if req is not None]
                if valid_requests:
                    # Merge reasoning and llm_kwargs
                    call_kwargs = {**self.llm_kwargs, "reasoning": self.reasoning, **kwargs}
                    batch_responses = self.model.batch_chat_completion(
                        requests=valid_requests, temperature=self.temperature, **call_kwargs
                    )
                else:
                    batch_responses = []

                # Check for failed responses and handle according to retry configuration
                failed_indices: List[int] = []
                for idx, response in enumerate(batch_responses):
                    if response.get("failed", False):
                        failed_indices.append(idx)

                if failed_indices:
                    logger.warning(
                        f"Batch processing had {len(failed_indices)} failed responses out of {len(batch_responses)}"
                    )

                    if not self.enable_retry:
                        # No retry allowed - fail immediately
                        first_failed = batch_responses[failed_indices[0]]
                        failed_content = first_failed.get("content", "Unknown error")
                        block_info = f"batch index {failed_indices[0]}"
                        raise GenerationFailedError(
                            message=f"Batch generation failed (retry disabled): {failed_content[:200]}",
                            block_info=block_info,
                        )

                    # Retry failed responses individually
                    logger.info(f"Retrying {len(failed_indices)} failed batch responses individually")
                    for failed_idx in failed_indices:
                        # Find the corresponding request
                        original_request = valid_requests[failed_idx]
                        header = original_request["metadata"].get("current_header", "unknown")[:50]
                        block_info = f"batch index {failed_idx}, header: {header}"

                        # Retry using the retry method
                        call_kwargs_retry = {**self.llm_kwargs, "reasoning": self.reasoning, **kwargs}
                        retried_content, retried_gen_id = self._make_llm_call_with_retry(
                            system_prompt=original_request["system_prompt"],
                            user_prompt=original_request["user_prompt"],
                            block_info=block_info,
                            **call_kwargs_retry,
                        )

                        # Update the response with the retried result
                        batch_responses[failed_idx] = {
                            "content": retried_content,
                            "generation_id": retried_gen_id,
                            "metadata": original_request["metadata"],
                            "failed": False,
                        }

                # Reconstruct results maintaining original order
                processed_blocks = []
                response_idx = 0
                for i, (current_block, original_block) in enumerate(batch):
                    if batch_requests[i] is None:
                        # This was a skipped block
                        current_header, current_body = self._get_header_and_body(block=current_block)
                        processed_blocks.append(f"{current_header}\n\n{current_body}\n\n")
                    else:
                        # Get the response for this block
                        response = batch_responses[response_idx]
                        response_idx += 1

                        request = batch_requests[i]
                        assert request is not None  # Type guard for mypy
                        request_dict: Dict[str, Any] = cast(Dict[str, Any], request)
                        metadata: Dict[str, Any] = cast(Dict[str, Any], request_dict.get("metadata", {}))
                        processed_body = response["content"]
                        generation_id = response.get("generation_id")

                        # Track generation ID for cost calculation
                        if generation_id:
                            add_generation_id(phase_name=self.name, generation_id=generation_id)

                        # Apply post-processing
                        processed_body = self._apply_post_processing(
                            original_block=metadata["current_body"], llm_block=processed_body, **kwargs
                        )

                        # Use the subclass-specific assembly method
                        processed_block = self._assemble_processed_block(
                            current_header=metadata["current_header"],
                            current_body=metadata["current_body"],
                            llm_response=processed_body,
                            original_body=metadata["original_body"],
                            **kwargs,
                        )
                        processed_blocks.append(processed_block)

                return processed_blocks

            else:
                # Fall back to sequential processing
                logger.debug("Model does not support batch processing, falling back to sequential processing")
                return self._process_batch_sequential(batch, **kwargs)

        except (GenerationFailedError, MaxRetriesExceededError):
            # Let our retry-related exceptions propagate to stop the pipeline
            raise
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            # Batch processing errors trigger fallback to sequential processing
            logger.warning(f"Batch processing failed: {str(e)}, falling back to sequential processing")
            return self._process_batch_sequential(batch, **kwargs)

    def _process_batch_sequential(self, batch: List[Tuple[str, str]], **kwargs) -> List[str]:
        """
        Process a batch of blocks sequentially (fallback when batch API is not available).

        Args:
            batch: List of tuples containing (current_block, original_block)
            **kwargs: Additional arguments to pass to the processing methods

        Returns:
            List of processed block strings
        """
        processed_blocks = []
        for current_block, original_block in batch:
            processed_block = self._process_block(current_block, original_block, **kwargs)
            processed_blocks.append(processed_block)
        return processed_blocks

    def _extract_blocks(self, text: str) -> List[str]:
        """
        Extract markdown blocks from text, including preamble.

        Args:
            text (str): The text to extract blocks from

        Returns:
            List[str]: List of extracted blocks
        """
        pattern = r"(?:^|\n)(#{1,6}\s+.*?)(?=\n#{1,6}\s+|\n*$)"
        matches = list(re.finditer(pattern, text, flags=re.DOTALL))

        blocks = []
        if not matches:
            if text.strip():
                blocks.append(text)
            return blocks

        # Check for preamble
        first_match = matches[0]
        if first_match.start() > 0:
            preamble = text[: first_match.start()]
            if preamble.strip():
                blocks.append(preamble)

        # Add matched blocks
        for match in matches:
            blocks.append(match.group(1))

        return blocks

    def _process_markdown_blocks(self, **kwargs) -> None:
        """
        Process all markdown blocks in the input text.

        This method identifies markdown blocks in the input text and processes
        them either in batches (if supported and enabled) or in parallel using
        the LLM model. It handles both current and original content to ensure
        proper processing.

        Args:
            **kwargs: Additional arguments to pass to the processing methods
        """
        try:
            # Pattern to match markdown headers and their content
            # Use a different approach that works better with re.DOTALL
            logger.info(f"Starting to process markdown blocks with {self.max_workers} workers")

            # Find all markdown blocks in both current and original text
            current_blocks = self._extract_blocks(self.input_text)
            original_blocks = self._extract_blocks(self.original_text)
            logger.info(f"Found {len(current_blocks)} current markdown blocks to process")
            logger.info(f"Found {len(original_blocks)} original markdown blocks")

            if not current_blocks:
                logger.warning("No markdown blocks found in the input text")
                self.content = self.input_text
                return
            if len(current_blocks) != len(original_blocks):
                msg = f"Block length mismatch: {len(current_blocks)} != {len(original_blocks)}"
                raise ValueError(msg)

            blocks = list(zip(current_blocks, original_blocks))

            # Check if we should use batch processing
            has_method = hasattr(self.model, "supports_batch")
            supports = self.model.supports_batch() if has_method else "N/A"
            logger.info(f"BATCH DEBUG: use_batch={self.use_batch}, has_method={has_method}, supports={supports}")
            if self.use_batch and hasattr(self.model, "supports_batch") and self.model.supports_batch():
                if self.batch_size:
                    # If batch_size is specified, process in chunks
                    logger.info(f"Using batch processing with batch_size={self.batch_size} for {len(blocks)} blocks")
                    processed_blocks = []
                    for i in tqdm(range(0, len(blocks), self.batch_size), desc=f"Processing {self.name} (batches)"):
                        batch = blocks[i : i + self.batch_size]
                        batch_results = self._process_batch(batch, **kwargs)
                        processed_blocks.extend(batch_results)
                else:
                    # Process all blocks at once in a single batch
                    logger.info(f"Using batch processing for all {len(blocks)} blocks at once")
                    processed_blocks = self._process_batch(blocks, **kwargs)

            else:
                # Process blocks in parallel (original behavior)
                logger.debug("Starting parallel block processing")

                func = partial(self._process_block, **kwargs)
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    processed_blocks = list(
                        tqdm(
                            iterable=executor.map(lambda args: func(*args), blocks),
                            total=len(blocks),
                            desc=f"Processing {self.name}",
                        )
                    )

            logger.debug(f"Successfully processed {len(processed_blocks)} blocks")
            logger.debug("Joining processed blocks into final content")
            self.content = "".join(processed_blocks)
            logger.info(f"Completed processing all blocks. Total output length: {len(self.content)} characters")

        except Exception as e:
            logger.error(f"Error during markdown block processing: {str(e)}")
            logger.exception("Stack trace for markdown block processing error")
            raise

    def run(self, **kwargs) -> None:
        """
        Execute the LLM phase.

        This method orchestrates the entire phase execution, including
        processing markdown blocks and writing the output file.

        Args:
            **kwargs: Additional arguments to pass to the processing methods
        """
        try:
            logger.info(f"Starting LLM phase: {self.name}")

            # Count tokens at the start
            self.start_token_count = self._count_tokens(self.input_text)
            logger.info(f"Phase '{self.name}' starting with ~{self.start_token_count:,} tokens")

            logger.debug("Processing markdown blocks")
            self._process_markdown_blocks(**kwargs)

            # Count tokens at the end
            self.end_token_count = self._count_tokens(self.content)
            logger.info(f"Phase '{self.name}' completed with ~{self.end_token_count:,} tokens")

            if self.use_subblocks:
                with self._subblock_stats_lock:
                    subblocks_processed_total = self._subblocks_processed_total
                    subblock_blocks_processed_total = self._subblock_blocks_processed_total
                    max_subblocks_in_single_block = self._max_subblocks_in_single_block

                logger.info(
                    "Sub-block processing summary for phase "
                    f"'{self.name}': processed {subblocks_processed_total} sub-blocks across "
                    f"{subblock_blocks_processed_total} blocks (max sub-blocks in a single block: "
                    f"{max_subblocks_in_single_block})"
                )

            logger.debug("Writing output file")
            self._write_output_file(content=self.content)
            logger.success(f"Successfully completed LLM phase: {self.name}")
        except Exception as e:
            logger.error(f"Failed to complete LLM phase {self.name}: {str(e)}")
            logger.exception("Stack trace for LLM phase error")
            raise


class StandardLlmPhase(LlmPhase):
    """
    Standard LLM phase for processing markdown content.

    This phase processes markdown blocks by sending them to an LLM model
    and applying post-processing to clean up the results.
    """

    def _assemble_processed_block(
        self, current_header: str, current_body: str, llm_response: str, original_body: str, **kwargs
    ) -> str:
        """
        Assemble the block by replacing the body with the LLM response.

        Args:
            current_header: The header of the current block
            current_body: The body of the current block (unused in standard phase)
            llm_response: The processed content from the LLM
            original_body: The original body for reference (unused in standard phase)
            **kwargs: Additional context

        Returns:
            The block with the LLM response as the body
        """
        # Standard behavior: replace the body with the LLM response
        if llm_response.strip():
            return f"{current_header}\n\n{llm_response}\n\n"
        else:
            return f"{current_header}\n\n"

    def _process_subblock(
        self,
        current_subblock: str,
        original_subblock: str,
        current_header: str,
        original_header: str,
        subblock_index: int,
        **kwargs,
    ) -> str:
        """
        Process a single sub-block using the LLM model.

        Args:
            current_subblock: The current sub-block text to process
            original_subblock: The original sub-block text for reference
            current_header: The header of the block (for context in prompts)
            original_header: The original header (for context in prompts)
            subblock_index: Index of the sub-block for logging
            **kwargs: Additional context or parameters

        Returns:
            str: The processed sub-block content
        """
        # Format the user message using sub-block content
        user_message = self._format_user_message(
            current_body=current_subblock,
            original_body=original_subblock,
            current_header=current_header,
            original_header=original_header,
        )

        # Get LLM response with retry logic
        call_kwargs = {**self.llm_kwargs, "reasoning": self.reasoning}
        block_info = f"header: {current_header[:50]}, subblock: {subblock_index}"
        processed_subblock, generation_id = self._make_llm_call_with_retry(
            system_prompt=self.system_prompt,
            user_prompt=user_message,
            block_info=block_info,
            **call_kwargs,
        )

        # Track generation ID for cost calculation
        add_generation_id(phase_name=self.name, generation_id=generation_id)

        return processed_subblock

    def _process_block_with_subblocks(self, current_block: str, original_block: str, **kwargs) -> str:
        """
        Process a markdown block by splitting it into sub-blocks and processing each.

        This method:
        1. Extracts the header and body
        2. Splits the body into paragraphs
        3. Groups paragraphs into token-bounded sub-blocks
        4. Processes each sub-block with the LLM (in parallel if max_workers > 1)
        5. Concatenates results and applies post-processing to the full body
        6. Reassembles the block with header

        Args:
            current_block (str): The current markdown block to process
            original_block (str): The original markdown block for reference
            **kwargs: Additional context or parameters

        Returns:
            str: The processed block content
        """
        # Extract header and body from both blocks
        current_header, current_body = self._get_header_and_body(block=current_block)
        original_header, original_body = self._get_header_and_body(block=original_block)

        # Check if there's any content to process (not empty and not just special tags)
        if (not current_body.strip() and not original_body.strip()) or (
            self._contains_only_special_tags(current_body) and self._contains_only_special_tags(original_body)
        ):
            logger.debug("Empty block content or content with only special tags, returning block as-is")
            return f"{current_header}\n\n{current_body}\n\n"

        # Split bodies into paragraphs
        current_paragraphs = self._split_body_into_paragraphs(current_body)
        original_paragraphs = self._split_body_into_paragraphs(original_body)

        # If no paragraphs, return as-is
        if not current_paragraphs:
            logger.debug("No paragraphs found in body, returning block as-is")
            return f"{current_header}\n\n{current_body}\n\n"

        # Group paragraphs into sub-blocks
        current_subblocks = self._group_paragraphs_into_subblocks(current_paragraphs)
        original_subblocks = self._group_paragraphs_into_subblocks(original_paragraphs)

        with self._subblock_stats_lock:
            self._subblock_blocks_processed_total += 1
            self._subblocks_processed_total += len(current_subblocks)
            if len(current_subblocks) > self._max_subblocks_in_single_block:
                self._max_subblocks_in_single_block = len(current_subblocks)

        logger.debug(f"Split block '{current_header[:30]}...' into {len(current_subblocks)} sub-blocks")

        # Ensure same number of sub-blocks (use original grouping if counts differ)
        if len(current_subblocks) != len(original_subblocks):
            logger.warning(
                f"Sub-block count mismatch: current={len(current_subblocks)}, original={len(original_subblocks)}. "
                "Using current body grouping for original as well."
            )
            # Re-group original using same number of paragraphs per group
            # This is a fallback - ideally counts should match
            original_subblocks = self._group_paragraphs_into_subblocks(original_paragraphs)
            if len(current_subblocks) != len(original_subblocks):
                # Last resort: pair by position, padding with empty strings
                while len(original_subblocks) < len(current_subblocks):
                    original_subblocks.append("")
                original_subblocks = original_subblocks[: len(current_subblocks)]

        # Process sub-blocks
        if self.max_workers > 1 and len(current_subblocks) > 1:
            # Parallel processing
            logger.debug(f"Processing {len(current_subblocks)} sub-blocks in parallel with {self.max_workers} workers")

            def process_subblock_wrapper(args: Tuple[int, str, str]) -> str:
                idx, curr_sb, orig_sb = args
                return self._process_subblock(
                    current_subblock=curr_sb,
                    original_subblock=orig_sb,
                    current_header=current_header,
                    original_header=original_header,
                    subblock_index=idx,
                    **kwargs,
                )

            subblock_args = [
                (i, curr_sb, orig_sb) for i, (curr_sb, orig_sb) in enumerate(zip(current_subblocks, original_subblocks))
            ]

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                processed_subblocks = list(executor.map(process_subblock_wrapper, subblock_args))
        else:
            # Sequential processing
            processed_subblocks = []
            for i, (curr_sb, orig_sb) in enumerate(zip(current_subblocks, original_subblocks)):
                processed_sb = self._process_subblock(
                    current_subblock=curr_sb,
                    original_subblock=orig_sb,
                    current_header=current_header,
                    original_header=original_header,
                    subblock_index=i,
                    **kwargs,
                )
                processed_subblocks.append(processed_sb)

        # Concatenate sub-block results with single newlines (matching input format)
        processed_body = "\n".join(processed_subblocks)

        # Apply post-processing to the full reassembled body
        processed_body = self._apply_post_processing(original_block=current_body, llm_block=processed_body, **kwargs)

        # Reconstruct the block
        if processed_body.strip():
            return f"{current_header}\n\n{processed_body}\n\n"
        logger.debug("Empty block body after sub-block processing, returning header only")
        return f"{current_header}\n\n"

    def _process_block(self, current_block: str, original_block: str, **kwargs) -> str:
        """
        Process a single markdown block using the LLM model.

        This method processes a markdown block by:
        1. Extracting the header and body
        2. Formatting a user message using the prompt template
        3. Sending the message to the LLM model (with retry if enabled)
        4. Applying post-processing to clean up the result

        If use_subblocks is enabled, delegates to _process_block_with_subblocks instead.

        Args:
            current_block (str): The current markdown block to process
            original_block (str): The original markdown block for reference
            **kwargs: Additional context or parameters

        Returns:
            str: The processed block content

        Raises:
            GenerationFailedError: If generation fails and retry is disabled
            MaxRetriesExceededError: If all retry attempts are exhausted
            EmptySectionError: If post-processing detects an invalid empty section
        """
        # Check if block should be skipped based on token count (before subblock processing)
        if self._should_skip_block_by_tokens(current_block):
            # Extract header and body to return block in proper format
            current_header, current_body = self._get_header_and_body(block=current_block)
            return f"{current_header}\n\n{current_body}\n\n"

        # Check if sub-block processing is enabled
        if self.use_subblocks:
            return self._process_block_with_subblocks(current_block, original_block, **kwargs)

        # Extract header and body from both blocks
        current_header, current_body = self._get_header_and_body(block=current_block)
        original_header, original_body = self._get_header_and_body(block=original_block)

        # Check if there's any content to process (not empty and not just special tags)
        if (not current_body.strip() and not original_body.strip()) or (
            self._contains_only_special_tags(current_body) and self._contains_only_special_tags(original_body)
        ):
            logger.debug("Empty block content or content with only special tags, returning block as-is")
            return f"{current_header}\n\n{current_body}\n\n"

        # Format the user message
        body = self._format_user_message(
            current_body=current_body,
            original_body=original_body,
            current_header=current_header,
            original_header=original_header,
        )

        # Get LLM response with retry logic
        # Merge reasoning and llm_kwargs
        call_kwargs = {**self.llm_kwargs, "reasoning": self.reasoning}
        block_info = f"header: {current_header[:50]}"
        processed_body, generation_id = self._make_llm_call_with_retry(
            system_prompt=self.system_prompt,
            user_prompt=body,
            block_info=block_info,
            **call_kwargs,
        )

        # Track generation ID for cost calculation
        add_generation_id(phase_name=self.name, generation_id=generation_id)

        # Apply post-processing
        processed_body = self._apply_post_processing(original_block=current_body, llm_block=processed_body, **kwargs)

        # Reconstruct the block
        if processed_body.strip():
            return f"{current_header}\n\n{processed_body}\n\n"
        logger.debug("Empty block body, returning header only")
        return f"{current_header}\n\n"


class IntroductionAnnotationPhase(LlmPhase):
    """
    LLM phase that adds introduction annotations to the beginning of each block.
    Uses the block content as input to generate an introduction, then prepends it to the block.
    """

    def _assemble_processed_block(
        self, current_header: str, current_body: str, llm_response: str, original_body: str, **kwargs
    ) -> str:
        """
        Assemble the block with the introduction prepended to the original content.

        Args:
            current_header: The header of the current block
            current_body: The body of the current block
            llm_response: The introduction generated by the LLM
            original_body: The original body for reference
            **kwargs: Additional context

        Returns:
            The block with introduction prepended
        """
        # Prepend the introduction to the current body
        return f"{current_header}\n\n{llm_response}\n\n{current_body}\n\n"

    def _process_block(self, current_block: str, original_block: str, **kwargs) -> str:
        """
        Process a single Markdown block by adding an introduction annotation at the beginning.

        Args:
            current_block (str): The markdown block currently being processed
                (may contain content from previous phases)
            original_block (str): The completely unedited block from the original text
            **kwargs: Additional arguments to pass to the chat completion

        Returns:
            str: The processed markdown block with introduction annotation

        Raises:
            GenerationFailedError: If generation fails and retry is disabled
            MaxRetriesExceededError: If all retry attempts are exhausted
            EmptySectionError: If post-processing detects an invalid empty section
        """
        # Check if block should be skipped based on token count (before processing)
        if self._should_skip_block_by_tokens(current_block):
            current_header, current_body = self._get_header_and_body(current_block)
            return f"{current_header}\n\n{current_body}\n\n"

        current_header, current_body = self._get_header_and_body(current_block)
        original_header, original_body = self._get_header_and_body(original_block)

        # Check if there's any content to process (not empty and not just special tags)
        if (not current_body.strip() and not original_body.strip()) or (
            self._contains_only_special_tags(current_body) and self._contains_only_special_tags(original_body)
        ):
            logger.debug("Empty block content or content with only special tags, returning block as-is")
            return f"{current_header}\n\n{current_body}\n\n"

        # Use the block content as the user prompt for generating the introduction
        user_prompt = self._format_user_message(current_body, original_body, current_header, original_header)

        if user_prompt:
            # Merge reasoning and llm_kwargs
            call_kwargs = {**self.llm_kwargs, "reasoning": self.reasoning, **kwargs}
            block_info = f"header: {current_header[:50]}"
            introduction, generation_id = self._make_llm_call_with_retry(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                block_info=block_info,
                temperature=self.temperature,
                **call_kwargs,
            )

            # Track generation ID for cost calculation
            add_generation_id(phase_name=self.name, generation_id=generation_id)

            # Apply post-processing to the introduction
            introduction = self._apply_post_processing(current_body, introduction, **kwargs)

            # Combine the introduction with the original block
            return f"{current_header}\n\n{introduction}\n\n{current_body}\n\n"
        else:
            logger.debug("Empty block body, returning header only")
            return f"{current_header}\n\n"


class SummaryAnnotationPhase(LlmPhase):
    """
    LLM phase that adds summary annotations to the end of each block.
    Uses the block content as input to generate a summary, then appends it to the block.
    """

    def _assemble_processed_block(
        self, current_header: str, current_body: str, llm_response: str, original_body: str, **kwargs
    ) -> str:
        """
        Assemble the block with the summary appended to the original content.

        Args:
            current_header: The header of the current block
            current_body: The body of the current block
            llm_response: The summary generated by the LLM
            original_body: The original body for reference
            **kwargs: Additional context

        Returns:
            The block with summary appended
        """
        # Append the summary to the current body
        return f"{current_header}\n\n{current_body}\n\n{llm_response}\n\n"

    def _process_block(self, current_block: str, original_block: str, **kwargs) -> str:
        """
        Process a single Markdown block by adding a summary annotation at the end.

        Args:
            current_block (str): The markdown block currently being processed
                (may contain content from previous phases)
            original_block (str): The completely unedited block from the original text
            **kwargs: Additional arguments to pass to the chat completion

        Returns:
            str: The processed markdown block with summary annotation

        Raises:
            GenerationFailedError: If generation fails and retry is disabled
            MaxRetriesExceededError: If all retry attempts are exhausted
            EmptySectionError: If post-processing detects an invalid empty section
        """
        # Check if block should be skipped based on token count (before processing)
        if self._should_skip_block_by_tokens(current_block):
            current_header, current_body = self._get_header_and_body(current_block)
            return f"{current_header}\n\n{current_body}\n\n"

        current_header, current_body = self._get_header_and_body(current_block)
        original_header, original_body = self._get_header_and_body(original_block)

        # Check if there's any content to process (not empty and not just special tags)
        if (not current_body.strip() and not original_body.strip()) or (
            self._contains_only_special_tags(current_body) and self._contains_only_special_tags(original_body)
        ):
            logger.debug("Empty block content or content with only special tags, returning block as-is")
            return f"{current_header}\n\n{current_body}\n\n"

        # Use the block content as the user prompt for generating the summary
        user_prompt = self._format_user_message(current_body, original_body, current_header, original_header)

        if user_prompt:
            # Merge reasoning and llm_kwargs
            call_kwargs = {**self.llm_kwargs, "reasoning": self.reasoning, **kwargs}
            block_info = f"header: {current_header[:50]}"
            summary, generation_id = self._make_llm_call_with_retry(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                block_info=block_info,
                temperature=self.temperature,
                **call_kwargs,
            )

            # Track generation ID for cost calculation
            add_generation_id(phase_name=self.name, generation_id=generation_id)

            # Apply post-processing to the summary
            summary = self._apply_post_processing(current_body, summary, **kwargs)

            # Combine the original block with the summary
            return f"{current_header}\n\n{current_body}\n\n{summary}\n\n"
        else:
            logger.debug("Empty block body, returning header only")
            return f"{current_header}\n\n"


class TwoStageFinalPhase(LlmPhase):
    """
    Two-stage FINAL phase that decomposes complex editorial work into:
    1. IDENTIFY: Analyze and list refinement opportunities (reasoning model)
    2. IMPLEMENT: Apply identified changes (cheaper model)

    To the end user, this behaves as a single phase. The two-stage
    decomposition is an internal implementation detail.

    Supports both batch and non-batch processing modes:
    - Non-batch: Per-block IDENTIFY → IMPLEMENT via _process_block(), blocks processed in parallel
    - Batch: Two-phase batching — batch all IDENTIFY calls first, then batch all IMPLEMENT calls
    """

    def __init__(
        self,
        name: str,
        input_file_path: Path,
        output_file_path: Path,
        original_file_path: Path,
        book_name: str,
        author_name: str,
        identify_model: LlmModel,
        implement_model: LlmModel,
        identify_system_prompt_path: Path,
        identify_user_prompt_path: Path,
        implement_system_prompt_path: Path,
        implement_user_prompt_path: Path,
        identify_temperature: float = LLM_DEFAULT_TEMPERATURE,
        implement_temperature: float = LLM_DEFAULT_TEMPERATURE,
        identify_reasoning: Optional[Dict[str, str]] = None,
        max_workers: Optional[int] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        post_processor_chain: Optional[PostProcessorChain] = None,
        use_batch: bool = False,
        batch_size: Optional[int] = None,
        enable_retry: bool = False,
        max_retries: int = DEFAULT_GENERATION_MAX_RETRIES,
        skip_if_less_than_tokens: Optional[int] = None,
    ) -> None:
        """
        Initialize the two-stage FINAL phase.

        Args:
            name: Name of the phase for logging and identification.
            input_file_path: Path to the input markdown file.
            output_file_path: Path where the processed output will be written.
            original_file_path: Path to the original markdown file for reference.
            book_name: Name of the book being processed.
            author_name: Name of the book's author.
            identify_model: LLM model for the IDENTIFY stage (analysis).
            implement_model: LLM model for the IMPLEMENT stage (application).
            identify_system_prompt_path: Path to the IDENTIFY system prompt.
            identify_user_prompt_path: Path to the IDENTIFY user prompt.
            implement_system_prompt_path: Path to the IMPLEMENT system prompt.
            implement_user_prompt_path: Path to the IMPLEMENT user prompt.
            identify_temperature: Temperature for the IDENTIFY stage.
            implement_temperature: Temperature for the IMPLEMENT stage.
            identify_reasoning: Optional reasoning configuration for IDENTIFY stage.
            max_workers: Maximum number of worker threads for parallel processing.
            llm_kwargs: Additional kwargs to pass to LLM calls.
            post_processor_chain: Chain of post-processors to apply to IMPLEMENT output.
            use_batch: Whether to use batch processing for LLM calls.
            batch_size: Number of items to process in each batch.
            enable_retry: Whether to retry failed generations.
            max_retries: Maximum number of retry attempts per generation.
            skip_if_less_than_tokens: If set, blocks with fewer tokens than this value
                will be skipped entirely (before chunking into subblocks). Default None means no skipping.
        """
        # Store two-stage specific attributes before calling parent __init__
        self.identify_model = identify_model
        self.implement_model = implement_model
        self.identify_system_prompt_path = identify_system_prompt_path
        self.identify_user_prompt_path = identify_user_prompt_path
        self.implement_system_prompt_path = implement_system_prompt_path
        self.implement_user_prompt_path = implement_user_prompt_path
        self.identify_temperature = identify_temperature
        self.implement_temperature = implement_temperature
        self.identify_reasoning = identify_reasoning or {}

        # Debug data collection (thread-safe)
        self._identify_debug_data: List[Dict[str, Any]] = []
        self._debug_data_lock = threading.Lock()

        # Store output directory for debug file
        self.output_dir = output_file_path.parent

        # Initialize content storage for prompts (will be loaded below)
        self.identify_system_prompt = ""
        self.identify_user_prompt = ""
        self.implement_system_prompt = ""
        self.implement_user_prompt = ""

        # Call parent __init__ with implement_model as primary (for compatibility)
        # We pass None for system_prompt_path and user_prompt_path since we handle prompts ourselves
        super().__init__(
            name=name,
            input_file_path=input_file_path,
            output_file_path=output_file_path,
            original_file_path=original_file_path,
            system_prompt_path=None,
            user_prompt_path=None,
            book_name=book_name,
            author_name=author_name,
            model=implement_model,  # Primary model for base class compatibility
            temperature=implement_temperature,
            max_workers=max_workers,
            reasoning=None,  # Reasoning is stage-specific
            llm_kwargs=llm_kwargs,
            post_processor_chain=post_processor_chain,
            use_batch=use_batch,
            batch_size=batch_size,
            enable_retry=enable_retry,
            max_retries=max_retries,
            skip_if_less_than_tokens=skip_if_less_than_tokens,
        )

        # Load stage-specific prompts
        self._load_stage_prompts()

        logger.info(
            f"TwoStageFinalPhase initialized with identify_model={identify_model}, implement_model={implement_model}"
        )

    def _load_stage_prompts(self) -> None:
        """Load and format prompts for both IDENTIFY and IMPLEMENT stages."""
        # Load IDENTIFY prompts
        self.identify_system_prompt = self._load_and_format_prompt(self.identify_system_prompt_path)
        self.identify_user_prompt = self._load_prompt_template(self.identify_user_prompt_path)

        # Load IMPLEMENT prompts
        self.implement_system_prompt = self._load_and_format_prompt(self.implement_system_prompt_path)
        self.implement_user_prompt = self._load_prompt_template(self.implement_user_prompt_path)

        logger.debug("Loaded all stage-specific prompts")

    def _load_and_format_prompt(self, prompt_path: Path) -> str:
        """Load a prompt file and format it with tags if needed."""
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with prompt_path.open(mode="r", encoding="utf-8") as f:
            content = f.read()

        # Format with tags_to_preserve if present
        format_params = {}

        # Add tags_to_preserve as format parameters
        tags_to_preserve = DEFAULT_TAGS_TO_PRESERVE
        if self.post_processor_chain:
            for processor in self.post_processor_chain.processors:
                if hasattr(processor, "tags_to_preserve"):
                    tags_to_preserve = processor.tags_to_preserve
                    break

        for tag in tags_to_preserve:
            tag_name = tag.strip("{}")
            format_params[tag_name] = tag

        if format_params:
            try:
                content = content.format(**format_params)
            except KeyError as e:
                logger.warning(f"Prompt contains undefined parameter: {e}")

        return content

    def _load_prompt_template(self, prompt_path: Path) -> str:
        """Load a prompt template file (without formatting)."""
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with prompt_path.open(mode="r", encoding="utf-8") as f:
            return f.read()

    def _read_system_prompt(self) -> str:
        """Override to return empty string (we handle prompts ourselves)."""
        return ""

    def _read_user_prompt(self) -> str:
        """Override to return empty string (we handle prompts ourselves)."""
        return ""

    def _format_identify_user_message(
        self,
        current_body: str,
        original_body: str,
        current_header: str,
    ) -> str:
        """Format the user message for the IDENTIFY stage."""
        context = {
            "current_body": current_body,
            "original_body": original_body,
            "current_header": current_header,
            "book_name": self.book_name,
            "author_name": self.author_name,
        }
        try:
            return self.identify_user_prompt.format(**context)
        except KeyError as e:
            logger.warning(f"IDENTIFY user prompt contains undefined parameter: {e}")
            return self.identify_user_prompt

    def _format_implement_user_message(
        self,
        current_body: str,
        current_header: str,
        changes: str,
    ) -> str:
        """Format the user message for the IMPLEMENT stage, including the changes list."""
        context = {
            "current_body": current_body,
            "current_header": current_header,
            "changes": changes,
            "book_name": self.book_name,
            "author_name": self.author_name,
        }
        try:
            return self.implement_user_prompt.format(**context)
        except KeyError as e:
            logger.warning(f"IMPLEMENT user prompt contains undefined parameter: {e}")
            return self.implement_user_prompt

    def _should_skip_block(self, current_body: str, original_body: str) -> bool:
        """Check if a block should be skipped (empty or only special tags)."""
        if not current_body.strip() and not original_body.strip():
            return True
        if self._contains_only_special_tags(current_body) and self._contains_only_special_tags(original_body):
            return True
        return False

    def _make_llm_call_with_model(
        self,
        model: LlmModel,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        reasoning: Optional[Dict[str, str]] = None,
        block_info: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Make an LLM call with a specific model (for stage-specific calls).

        Args:
            model: The LLM model to use for this call.
            system_prompt: The system prompt.
            user_prompt: The user prompt.
            temperature: Temperature setting for this call.
            reasoning: Optional reasoning configuration.
            block_info: Optional identifier for error messages.

        Returns:
            Tuple of (response_content, generation_id).
        """
        last_error: Optional[Exception] = None
        last_content: str = ""
        max_attempts = (self.max_retries + 1) if self.enable_retry else 1

        call_kwargs = {**self.llm_kwargs}
        if reasoning:
            call_kwargs["reasoning"] = reasoning

        for attempt in range(max_attempts):
            try:
                content, generation_id = model.chat_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    **call_kwargs,
                )

                if is_failed_response(content):
                    last_content = content
                    error_msg = f"LLM returned failed response: {content[:100]}..."
                    if self.enable_retry and attempt < self.max_retries:
                        logger.warning(
                            f"Generation failed (attempt {attempt + 1}/{max_attempts}): {error_msg}. Retrying..."
                        )
                        continue
                    else:
                        if self.enable_retry:
                            raise MaxRetriesExceededError(
                                message=f"Generation failed after {max_attempts} attempts: {error_msg}",
                                attempts=max_attempts,
                                block_info=block_info,
                            )
                        else:
                            raise GenerationFailedError(message=error_msg, block_info=block_info)

                return content, generation_id

            except (GenerationFailedError, MaxRetriesExceededError):
                raise
            except ResponseTruncatedError as e:
                last_error = e
                error_msg = f"Response truncated: {str(e)}"
                if self.enable_retry and attempt < self.max_retries:
                    logger.warning(f"Response truncated (attempt {attempt + 1}/{max_attempts}). Retrying...")
                    continue
                else:
                    if self.enable_retry:
                        raise MaxRetriesExceededError(
                            message=f"Response truncated after {max_attempts} attempts",
                            attempts=max_attempts,
                            block_info=block_info,
                        ) from e
                    else:
                        raise GenerationFailedError(message=error_msg, block_info=block_info) from e
            except Exception as e:
                last_error = e
                error_msg = f"LLM call failed: {str(e)}"
                if self.enable_retry and attempt < self.max_retries:
                    logger.warning(
                        f"Generation failed (attempt {attempt + 1}/{max_attempts}): {error_msg}. Retrying..."
                    )
                    continue
                else:
                    if self.enable_retry:
                        raise MaxRetriesExceededError(
                            message=f"Generation failed after {max_attempts} attempts: {error_msg}",
                            attempts=max_attempts,
                            block_info=block_info,
                        ) from e
                    else:
                        raise GenerationFailedError(message=error_msg, block_info=block_info) from e

        # Fallback (should not reach here)
        if last_error:
            raise GenerationFailedError(
                message=f"LLM call failed: {str(last_error)}", block_info=block_info
            ) from last_error
        else:
            raise GenerationFailedError(
                message=f"LLM returned failed response: {last_content[:100]}...", block_info=block_info
            )

    def _add_debug_data(
        self,
        block_index: int,
        block_header: str,
        identify_response: str,
        generation_id: Optional[str],
    ) -> None:
        """Thread-safe addition of debug data."""
        with self._debug_data_lock:
            self._identify_debug_data.append(
                {
                    "block_index": block_index,
                    "block_header": block_header,
                    "identify_response": identify_response,
                    "generation_id": generation_id,
                }
            )

    def _process_block(self, current_block: str, original_block: str, **kwargs) -> str:
        """
        Process a single markdown block through both IDENTIFY and IMPLEMENT stages.

        This method is used in non-batch mode for parallel processing.

        Args:
            current_block: The current markdown block to process.
            original_block: The original markdown block for reference.
            **kwargs: Additional context or parameters.

        Returns:
            The processed block content.
        """
        # Check if block should be skipped based on token count (before processing)
        if self._should_skip_block_by_tokens(current_block):
            current_header, current_body = self._get_header_and_body(block=current_block)
            return f"{current_header}\n\n{current_body}\n\n"

        current_header, current_body = self._get_header_and_body(block=current_block)
        original_header, original_body = self._get_header_and_body(block=original_block)

        # Skip empty/special-tag-only blocks
        if self._should_skip_block(current_body, original_body):
            logger.debug(f"Skipping block (empty or special tags only): {current_header[:50]}")
            return f"{current_header}\n\n{current_body}\n\n"

        # Get block index from kwargs if available
        block_index = kwargs.get("block_index", len(self._identify_debug_data))

        # Stage 1: IDENTIFY - Analyze and produce change list
        identify_user_message = self._format_identify_user_message(
            current_body=current_body,
            original_body=original_body,
            current_header=current_header,
        )

        changes_list, identify_gen_id = self._make_llm_call_with_model(
            model=self.identify_model,
            system_prompt=self.identify_system_prompt,
            user_prompt=identify_user_message,
            temperature=self.identify_temperature,
            reasoning=self.identify_reasoning,
            block_info=f"identify: {current_header[:50]}",
        )

        # Track generation ID for cost calculation
        add_generation_id(phase_name=f"{self.name}_identify", generation_id=identify_gen_id)

        # Store debug data (thread-safe)
        self._add_debug_data(
            block_index=block_index,
            block_header=current_header,
            identify_response=changes_list,
            generation_id=identify_gen_id,
        )

        # Stage 2: IMPLEMENT - Apply the identified changes
        implement_user_message = self._format_implement_user_message(
            current_body=current_body,
            current_header=current_header,
            changes=changes_list,
        )

        processed_body, implement_gen_id = self._make_llm_call_with_model(
            model=self.implement_model,
            system_prompt=self.implement_system_prompt,
            user_prompt=implement_user_message,
            temperature=self.implement_temperature,
            block_info=f"implement: {current_header[:50]}",
        )

        # Track generation ID for cost calculation
        add_generation_id(phase_name=f"{self.name}_implement", generation_id=implement_gen_id)

        # Apply post-processing to final output
        processed_body = self._apply_post_processing(
            original_block=current_body,
            llm_block=processed_body,
            **kwargs,
        )

        # Assemble final block
        if processed_body.strip():
            return f"{current_header}\n\n{processed_body}\n\n"
        return f"{current_header}\n\n"

    def _process_markdown_blocks(self, **kwargs) -> None:
        """
        Process all markdown blocks with two-phase batching when batch mode is enabled.

        This method overrides the base class to support the two-stage processing flow.
        """
        try:
            logger.info(f"Starting to process markdown blocks with {self.max_workers} workers")

            current_blocks = self._extract_blocks(self.input_text)
            original_blocks = self._extract_blocks(self.original_text)
            logger.info(f"Found {len(current_blocks)} current markdown blocks to process")
            logger.info(f"Found {len(original_blocks)} original markdown blocks")

            if not current_blocks:
                logger.warning("No markdown blocks found in the input text")
                self.content = self.input_text
                return

            if len(current_blocks) != len(original_blocks):
                msg = f"Block length mismatch: {len(current_blocks)} != {len(original_blocks)}"
                raise ValueError(msg)

            blocks = list(zip(current_blocks, original_blocks))

            # Check if batch mode is enabled and supported by BOTH models
            use_batch = (
                self.use_batch
                and hasattr(self.identify_model, "supports_batch")
                and self.identify_model.supports_batch()
                and hasattr(self.implement_model, "supports_batch")
                and self.implement_model.supports_batch()
            )

            logger.info(f"Two-stage FINAL: use_batch={use_batch}")

            if use_batch:
                # Two-phase batch processing
                if self.batch_size:
                    logger.info(f"Using batch processing with batch_size={self.batch_size}")
                    processed_blocks = []
                    for i in tqdm(range(0, len(blocks), self.batch_size), desc=f"Processing {self.name} (batches)"):
                        chunk = blocks[i : i + self.batch_size]
                        chunk_start_index = i
                        chunk_results = self._process_blocks_batch_mode(chunk, chunk_start_index, **kwargs)
                        processed_blocks.extend(chunk_results)
                else:
                    logger.info("Using batch processing for all blocks at once")
                    processed_blocks = self._process_blocks_batch_mode(blocks, 0, **kwargs)
            else:
                # Non-batch: parallel processing via _process_block()
                logger.debug("Starting parallel block processing (non-batch mode)")
                processed_blocks = self._process_blocks_parallel_mode(blocks, **kwargs)

            logger.debug(f"Successfully processed {len(processed_blocks)} blocks")
            self.content = "".join(processed_blocks)
            logger.info(f"Completed processing all blocks. Total output length: {len(self.content)} characters")

        except Exception as e:
            logger.error(f"Error during markdown block processing: {str(e)}")
            logger.exception("Stack trace for markdown block processing error")
            raise

    def _process_blocks_parallel_mode(self, blocks: List[Tuple[str, str]], **kwargs) -> List[str]:
        """Process blocks in parallel using ThreadPoolExecutor (non-batch mode)."""

        def process_with_index(args: Tuple[int, Tuple[str, str]]) -> str:
            idx, (current_block, original_block) = args
            return self._process_block(current_block, original_block, block_index=idx, **kwargs)

        indexed_blocks = list(enumerate(blocks))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            processed_blocks = list(
                tqdm(
                    iterable=executor.map(process_with_index, indexed_blocks),
                    total=len(blocks),
                    desc=f"Processing {self.name}",
                )
            )

        return processed_blocks

    def _process_blocks_batch_mode(
        self,
        blocks: List[Tuple[str, str]],
        start_index: int,
        **kwargs,
    ) -> List[str]:
        """
        Process all blocks using two-phase batch API calls.

        Phase 1: Batch all IDENTIFY calls → collect all change lists
        Phase 2: Batch all IMPLEMENT calls with respective change lists

        Args:
            blocks: List of (current_block, original_block) tuples.
            start_index: Starting index for block numbering (for batch_size chunking).
            **kwargs: Additional context.

        Returns:
            List of processed block strings.
        """
        # Prepare block data
        block_data = []
        for i, (current_block, original_block) in enumerate(blocks):
            current_header, current_body = self._get_header_and_body(block=current_block)
            original_header, original_body = self._get_header_and_body(block=original_block)
            # Skip if token count is below threshold or if block is empty/special tags only
            skip_by_tokens = self._should_skip_block_by_tokens(current_block)
            skip_by_content = self._should_skip_block(current_body, original_body)
            block_data.append(
                {
                    "index": start_index + i,
                    "current_header": current_header,
                    "current_body": current_body,
                    "original_body": original_body,
                    "skip": skip_by_tokens or skip_by_content,
                }
            )

        # Phase 1: Batch IDENTIFY
        logger.info(f"Phase 1: Running IDENTIFY batch for {len(block_data)} blocks")
        identify_results = self._run_identify_batch(block_data, **kwargs)

        # Store debug data for non-skipped blocks
        for bd, id_result in zip(block_data, identify_results):
            if not bd["skip"] and not id_result.get("skipped"):
                block_index: int = cast(int, bd["index"])
                self._add_debug_data(
                    block_index=block_index,
                    block_header=str(bd["current_header"]),
                    identify_response=id_result.get("content", ""),
                    generation_id=id_result.get("generation_id"),
                )

        # Phase 2: Batch IMPLEMENT (with change lists from Phase 1)
        logger.info(f"Phase 2: Running IMPLEMENT batch for {len(block_data)} blocks")
        implement_results = self._run_implement_batch(block_data, identify_results, **kwargs)

        # Assemble final blocks
        processed_blocks = []
        for bd, impl_result in zip(block_data, implement_results):
            if bd["skip"] or impl_result.get("skipped"):
                processed_blocks.append(f"{bd['current_header']}\n\n{bd['current_body']}\n\n")
            else:
                processed_body = self._apply_post_processing(
                    original_block=str(bd["current_body"]),
                    llm_block=impl_result.get("content", ""),
                    **kwargs,
                )
                if processed_body.strip():
                    processed_blocks.append(f"{bd['current_header']}\n\n{processed_body}\n\n")
                else:
                    processed_blocks.append(f"{bd['current_header']}\n\n")

        return processed_blocks

    def _run_identify_batch(self, block_data: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Run IDENTIFY stage as a batch API call."""
        requests: List[Optional[Dict[str, Any]]] = []
        for bd in block_data:
            if bd["skip"]:
                requests.append(None)  # Placeholder for skipped blocks
            else:
                user_message = self._format_identify_user_message(
                    current_body=bd["current_body"],
                    original_body=bd["original_body"],
                    current_header=bd["current_header"],
                )
                requests.append(
                    {
                        "system_prompt": self.identify_system_prompt,
                        "user_prompt": user_message,
                        "metadata": {"index": bd["index"]},
                    }
                )

        # Filter non-None requests and call batch API
        valid_requests = [r for r in requests if r is not None]
        if valid_requests:
            call_kwargs = {**self.llm_kwargs}
            if self.identify_reasoning:
                call_kwargs["reasoning"] = self.identify_reasoning
            batch_responses = self.identify_model.batch_chat_completion(
                requests=valid_requests,
                temperature=self.identify_temperature,
                **call_kwargs,
            )

            # Track generation IDs
            for response in batch_responses:
                gen_id = response.get("generation_id")
                if gen_id:
                    add_generation_id(phase_name=f"{self.name}_identify", generation_id=gen_id)
        else:
            batch_responses = []

        # Handle failed responses
        self._handle_failed_batch_responses(batch_responses, valid_requests, "identify")

        # Reconstruct results with placeholders for skipped blocks
        results: List[Dict[str, Any]] = []
        response_idx = 0
        for req in requests:
            if req is None:
                results.append({"content": "", "skipped": True})
            else:
                results.append(batch_responses[response_idx])
                response_idx += 1

        return results

    def _run_implement_batch(
        self,
        block_data: List[Dict[str, Any]],
        identify_results: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Run IMPLEMENT stage as a batch API call with IDENTIFY results."""
        requests: List[Optional[Dict[str, Any]]] = []
        for bd, id_result in zip(block_data, identify_results):
            if bd["skip"] or id_result.get("skipped"):
                requests.append(None)
            else:
                user_message = self._format_implement_user_message(
                    current_body=bd["current_body"],
                    current_header=bd["current_header"],
                    changes=id_result.get("content", ""),
                )
                requests.append(
                    {
                        "system_prompt": self.implement_system_prompt,
                        "user_prompt": user_message,
                        "metadata": {"index": bd["index"]},
                    }
                )

        # Filter and call batch API
        valid_requests = [r for r in requests if r is not None]
        if valid_requests:
            batch_responses = self.implement_model.batch_chat_completion(
                requests=valid_requests,
                temperature=self.implement_temperature,
                **self.llm_kwargs,
            )

            # Track generation IDs
            for response in batch_responses:
                gen_id = response.get("generation_id")
                if gen_id:
                    add_generation_id(phase_name=f"{self.name}_implement", generation_id=gen_id)
        else:
            batch_responses = []

        # Handle failed responses
        self._handle_failed_batch_responses(batch_responses, valid_requests, "implement")

        # Reconstruct results
        results: List[Dict[str, Any]] = []
        response_idx = 0
        for req in requests:
            if req is None:
                results.append({"content": "", "skipped": True})
            else:
                results.append(batch_responses[response_idx])
                response_idx += 1

        return results

    def _handle_failed_batch_responses(
        self,
        batch_responses: List[Dict[str, Any]],
        valid_requests: List[Dict[str, Any]],
        stage: str,
    ) -> None:
        """Handle failed responses in batch processing."""
        failed_indices = [idx for idx, response in enumerate(batch_responses) if response.get("failed", False)]

        if not failed_indices:
            return

        logger.warning(
            f"{stage.upper()} batch had {len(failed_indices)} failed responses out of {len(batch_responses)}"
        )

        if not self.enable_retry:
            first_failed = batch_responses[failed_indices[0]]
            failed_content = first_failed.get("content", "Unknown error")
            raise GenerationFailedError(
                message=f"{stage.upper()} batch generation failed (retry disabled): {failed_content[:200]}",
                block_info=f"batch index {failed_indices[0]}",
            )

        # Retry failed responses individually
        logger.info(f"Retrying {len(failed_indices)} failed {stage} responses individually")
        model = self.identify_model if stage == "identify" else self.implement_model
        temperature = self.identify_temperature if stage == "identify" else self.implement_temperature
        reasoning = self.identify_reasoning if stage == "identify" else None

        for failed_idx in failed_indices:
            original_request = valid_requests[failed_idx]
            block_info = f"{stage} batch index {failed_idx}"

            retried_content, retried_gen_id = self._make_llm_call_with_model(
                model=model,
                system_prompt=original_request["system_prompt"],
                user_prompt=original_request["user_prompt"],
                temperature=temperature,
                reasoning=reasoning,
                block_info=block_info,
            )

            # Update the response
            batch_responses[failed_idx] = {
                "content": retried_content,
                "generation_id": retried_gen_id,
                "metadata": original_request.get("metadata"),
                "failed": False,
            }

    def _write_identify_debug_file(self) -> None:
        """Write IDENTIFY stage responses to a debug JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = self.output_dir / f"final_identify_debug_{timestamp}.json"

        # Sort debug data by block index
        sorted_debug_data = sorted(self._identify_debug_data, key=lambda x: x.get("block_index", 0))

        debug_data = {
            "timestamp": datetime.now().isoformat(),
            "phase_name": self.name,
            "book_name": self.book_name,
            "author_name": self.author_name,
            "identify_model": str(self.identify_model),
            "implement_model": str(self.implement_model),
            "blocks": sorted_debug_data,
        }

        try:
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)
            logger.info(f"IDENTIFY debug output written to: {debug_file}")
        except Exception as e:
            logger.error(f"Failed to write IDENTIFY debug file: {e}")

    def run(self, **kwargs) -> None:
        """
        Execute the two-stage FINAL phase.

        Overrides base class to write debug JSON file after processing.
        """
        try:
            logger.info(f"Starting two-stage FINAL phase: {self.name}")

            # Count tokens at the start
            self.start_token_count = self._count_tokens(self.input_text)
            logger.info(f"Phase '{self.name}' starting with ~{self.start_token_count:,} tokens")

            # Process markdown blocks
            self._process_markdown_blocks(**kwargs)

            # Count tokens at the end
            self.end_token_count = self._count_tokens(self.content)
            logger.info(f"Phase '{self.name}' completed with ~{self.end_token_count:,} tokens")

            # Write output file
            self._write_output_file(content=self.content)

            # Write debug file with IDENTIFY outputs
            self._write_identify_debug_file()

            logger.success(f"Successfully completed two-stage FINAL phase: {self.name}")
        except Exception as e:
            logger.error(f"Failed to complete two-stage FINAL phase {self.name}: {str(e)}")
            logger.exception("Stack trace for phase error")
            raise
