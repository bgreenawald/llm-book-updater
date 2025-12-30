import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from llm_core import (
    GenerationFailedError,
    LlmModel,
    MaxRetriesExceededError,
)
from llm_core.config import (
    DEFAULT_GENERATION_MAX_RETRIES,
    DEFAULT_MAX_SUBBLOCK_TOKENS,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_SUBBLOCK_TOKENS,
    DEFAULT_TAGS_TO_PRESERVE,
)
from llm_core.cost import add_generation_id
from loguru import logger
from tqdm import tqdm

from book_updater.phases.utils import (
    TokenCounter,
    contains_only_special_tags,
    extract_markdown_blocks,
    get_header_and_body,
    make_llm_call_with_retry,
    map_batch_responses_to_requests,
    read_file,
    should_skip_by_token_count,
    write_file,
)
from book_updater.processing.post_processors import EmptySectionError, PostProcessorChain


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
        self._token_counter = TokenCounter()  # Use shared utility for token counting

        # Initialize logging
        logger.info(f"Initializing LlmPhase: {name}")
        logger.debug(f"Input file: {input_file_path}")
        logger.debug(f"Original file: {original_file_path}")
        logger.debug(f"Output file: {output_file_path}")
        logger.debug(f"System prompt path: {system_prompt_path}")
        logger.debug(f"User prompt path: {user_prompt_path}")
        logger.debug(f"Book: {book_name} by {author_name}")
        logger.debug(f"Max workers: {max_workers}")
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
        return self._token_counter.count(text)

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
        except Exception as e:
            # All post-processor errors should stop the pipeline to prevent corrupt output
            logger.error(f"Post-processing failed: {str(e)}")
            logger.exception("Post-processing error stack trace")
            raise RuntimeError(
                f"Post-processing failed for block. This is a critical error that prevents "
                f"producing correct output. Original error: {str(e)}"
            ) from e

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
        # Delegate to shared utility function
        return make_llm_call_with_retry(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            enable_retry=self.enable_retry,
            max_retries=self.max_retries,
            block_info=block_info,
            **kwargs,
        )

    def _read_input_file(self) -> str:
        """Read the input markdown file."""
        return read_file(self.input_file_path)

    def _read_original_file(self) -> str:
        """Read the original markdown file."""
        return read_file(self.original_file_path)

    def _write_output_file(self, content: str) -> None:
        """Write processed content to output file."""
        write_file(self.output_file_path, content)
        logger.info(f"Successfully wrote output to: {self.output_file_path}")

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

            content = read_file(self.system_prompt_path)
            logger.debug(f"Read system prompt from {self.system_prompt_path}")

            # Format the system prompt with parameters if needed
            format_params = {}

            # Get tags to preserve from post-processor chain if available
            tags_to_preserve = self._get_tags_to_preserve()

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
        """Read the user prompt file."""
        if not self.user_prompt_path or not self.user_prompt_path.exists():
            raise FileNotFoundError(f"User prompt file not found: {self.user_prompt_path}")
        return read_file(self.user_prompt_path)

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
            raise ValueError(
                f"User prompt contains undefined parameter: {e}. Available context keys: {list(context.keys())}"
            ) from e

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
        # Delegate to shared utility function
        return get_header_and_body(block)

    def _get_tags_to_preserve(self) -> List[str]:
        """Return the configured tags to preserve for this phase."""
        tags_to_preserve = list(DEFAULT_TAGS_TO_PRESERVE)
        if self.post_processor_chain:
            for processor in self.post_processor_chain.processors:
                if hasattr(processor, "tags_to_preserve"):
                    tags_to_preserve = processor.tags_to_preserve
                    break
        return tags_to_preserve

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
        For example, AnnotationPhase can prepend or append the LLM response.

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
            if not (hasattr(self.model, "supports_batch") and self.model.supports_batch()):
                logger.debug("Model doesn't support batch, using sequential")
                return self._process_batch_sequential(batch, **kwargs)

            logger.debug(f"Processing batch of {len(batch)} blocks using batch API")

            if self.use_subblocks:
                return self._process_batch_with_subblocks(batch, **kwargs)
            return self._process_batch_standard(batch, **kwargs)

        except (GenerationFailedError, MaxRetriesExceededError):
            # Let our retry-related exceptions propagate to stop the pipeline
            raise
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            # Batch processing errors trigger fallback to sequential processing
            logger.warning(f"Batch processing failed: {e}, using sequential")
            return self._process_batch_sequential(batch, **kwargs)

    def _process_batch_with_subblocks(self, batch: List[Tuple[str, str]], **kwargs) -> List[str]:
        block_entries: List[Dict[str, Any]] = []
        valid_requests: List[Dict[str, Any]] = []
        tags_to_preserve = self._get_tags_to_preserve()

        for block_index, (current_block, original_block) in enumerate(batch):
            current_header, current_body = self._get_header_and_body(block=current_block)
            original_header, original_body = self._get_header_and_body(block=original_block)

            # Skip blocks based on token count threshold (before subblock processing)
            if should_skip_by_token_count(current_block, self.skip_if_less_than_tokens, self._token_counter):
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
                contains_only_special_tags(current_body, tags_to_preserve)
                and contains_only_special_tags(original_body, tags_to_preserve)
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

            with self._subblock_stats_lock:
                self._subblock_blocks_processed_total += 1
                self._subblocks_processed_total += len(current_subblocks)
                if len(current_subblocks) > self._max_subblocks_in_single_block:
                    self._max_subblocks_in_single_block = len(current_subblocks)

            logger.debug(f"Batch mode: split block '{current_header[:30]}...' into {len(current_subblocks)} sub-blocks")

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

            for subblock_index, curr_sb in enumerate(current_subblocks):
                user_message = self._format_user_message(
                    current_body=curr_sb,
                    original_body=original_body,
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
            batch_responses = self.model.batch_chat_completion(requests=valid_requests, **call_kwargs)
        else:
            batch_responses = []

        request_by_metadata: Dict[Any, Dict[str, Any]] = {}
        for req in valid_requests:
            subblock_req_dict: Dict[str, Any] = cast(Dict[str, Any], req)
            subblock_req_metadata: Dict[str, Any] = cast(Dict[str, Any], subblock_req_dict.get("metadata", {}))
            block_idx = subblock_req_metadata.get("block_index")
            subblock_idx = subblock_req_metadata.get("subblock_index")
            if block_idx is not None and subblock_idx is not None:
                request_by_metadata[(int(block_idx), int(subblock_idx))] = req

        self._retry_failed_batch_responses(
            batch_responses=batch_responses,
            request_lookup=request_by_metadata,
            stage_name="subblock",
            **kwargs,
        )

        # Group responses back into blocks using metadata-based mapping
        # This handles cases where providers return results out of order
        # Batch API should preserve metadata from requests to responses
        subblock_outputs: Dict[int, List[Optional[str]]] = {}
        for resp in batch_responses:
            resp_metadata: Dict[str, Any] = cast(Dict[str, Any], resp.get("metadata", {}))
            block_index = int(resp_metadata.get("block_index", -1))
            subblock_index = int(resp_metadata.get("subblock_index", -1))
            subblock_count = int(resp_metadata.get("subblock_count", 1))

            if block_index < 0 or subblock_index < 0:
                logger.error(
                    f"Invalid metadata in batch response: {resp_metadata}. "
                    "Batch API should preserve metadata from requests."
                )
                raise ValueError(
                    f"Invalid metadata in batch response: {resp_metadata}. "
                    "Cannot reconstruct output safely. This indicates a problem with the batch API "
                    "preserving metadata from requests to responses."
                )

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

    def _process_batch_standard(self, batch: List[Tuple[str, str]], **kwargs) -> List[str]:
        # Prepare batch requests (non-subblock batching: one request per block)
        batch_requests: List[Optional[Dict[str, Any]]] = []
        tags_to_preserve = self._get_tags_to_preserve()
        for block_index, (current_block, original_block) in enumerate(batch):
            current_header, current_body = self._get_header_and_body(block=current_block)
            original_header, original_body = self._get_header_and_body(block=original_block)

            # Skip blocks based on token count threshold (before processing)
            if should_skip_by_token_count(current_block, self.skip_if_less_than_tokens, self._token_counter):
                batch_requests.append(None)  # Mark as skip
                continue

            # Skip empty blocks or blocks with only special tags
            if (not current_body.strip() and not original_body.strip()) or (
                contains_only_special_tags(current_body, tags_to_preserve)
                and contains_only_special_tags(original_body, tags_to_preserve)
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
                        "block_index": block_index,
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
            batch_responses = self.model.batch_chat_completion(requests=valid_requests, **call_kwargs)
        else:
            batch_responses = []

        # Map responses back to requests using metadata (handles out-of-order responses)
        mapped_responses = map_batch_responses_to_requests(
            batch_responses=batch_responses, requests=batch_requests, stage_name="non-subblock"
        )

        request_by_index: Dict[Any, Dict[str, Any]] = {}
        for req in valid_requests:
            non_subblock_req_dict: Dict[str, Any] = cast(Dict[str, Any], req)
            non_subblock_req_metadata: Dict[str, Any] = cast(Dict[str, Any], non_subblock_req_dict.get("metadata", {}))
            block_idx = non_subblock_req_metadata.get("block_index")
            if block_idx is not None:
                request_by_index[int(block_idx)] = req

        self._retry_failed_batch_responses(
            batch_responses=mapped_responses,
            request_lookup=request_by_index,
            stage_name="non-subblock",
            **kwargs,
        )

        # Reconstruct results maintaining original order
        processed_blocks = []
        for i, (current_block, original_block) in enumerate(batch):
            if batch_requests[i] is None:
                # This was a skipped block
                current_header, current_body = self._get_header_and_body(block=current_block)
                processed_blocks.append(f"{current_header}\n\n{current_body}\n\n")
            else:
                # Get the mapped response for this block
                response = mapped_responses[i]

                # Skip if this was a skipped response placeholder
                if response.get("skipped", False):
                    current_header, current_body = self._get_header_and_body(block=current_block)
                    processed_blocks.append(f"{current_header}\n\n{current_body}\n\n")
                    continue

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

    def _retry_failed_batch_responses(
        self,
        batch_responses: List[Dict[str, Any]],
        request_lookup: Dict[Any, Dict[str, Any]],
        stage_name: str,
        **kwargs,
    ) -> None:
        """Retry failed batch responses individually.

        Modifies batch_responses in-place, replacing failed responses with
        retry results while preserving metadata.
        """
        failed_responses = [(idx, resp) for idx, resp in enumerate(batch_responses) if resp.get("failed", False)]

        if not failed_responses:
            return

        logger.warning(f"{stage_name} batch had {len(failed_responses)} failures out of {len(batch_responses)}")

        if not self.enable_retry:
            _, first_failed = failed_responses[0]
            failed_content = first_failed.get("content", "Unknown error")
            failed_metadata = first_failed.get("metadata", {})
            block_index = failed_metadata.get("block_index", "unknown")
            subblock_index = failed_metadata.get("subblock_index")
            if subblock_index is not None:
                block_info = f"block index {block_index}, subblock {subblock_index}"
            else:
                block_info = f"block index {block_index}"
            raise GenerationFailedError(
                message=f"Batch generation failed (retry disabled): {failed_content[:200]}",
                block_info=block_info,
            )

        logger.info(f"Retrying {len(failed_responses)} failed responses individually")

        for resp_idx, failed_response in failed_responses:
            failed_metadata = failed_response.get("metadata", {})
            block_idx = failed_metadata.get("block_index")
            subblock_idx = failed_metadata.get("subblock_index")

            if block_idx is None:
                logger.error(f"Failed response at {resp_idx} missing block_index")
                continue

            lookup_key: Any
            block_info = f"block index {block_idx}"
            if subblock_idx is not None:
                lookup_key = (int(block_idx), int(subblock_idx))
                block_info = f"{block_info}, subblock {subblock_idx}"
            else:
                lookup_key = int(block_idx)

            original_request = request_lookup.get(lookup_key)
            if original_request is None:
                logger.error(f"No request for block_index={block_idx}")
                continue

            header = original_request["metadata"].get("current_header", "unknown")[:50]
            block_info = f"{block_info}, header: {header}"

            call_kwargs = {**self.llm_kwargs, "reasoning": self.reasoning, **kwargs}
            retried_content, retried_gen_id = self._make_llm_call_with_retry(
                system_prompt=original_request["system_prompt"],
                user_prompt=original_request["user_prompt"],
                block_info=block_info,
                **call_kwargs,
            )

            batch_responses[resp_idx] = {
                "content": retried_content,
                "generation_id": retried_gen_id,
                "metadata": failed_metadata,
                "failed": False,
            }

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
        # Delegate to shared utility function
        return extract_markdown_blocks(text)

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
