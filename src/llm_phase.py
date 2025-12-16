import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import tiktoken
from loguru import logger
from tqdm import tqdm

from src.constants import DEFAULT_MAX_WORKERS, DEFAULT_TAGS_TO_PRESERVE
from src.cost_tracking_wrapper import add_generation_id
from src.llm_model import LlmModel
from src.post_processors import PostProcessorChain


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
        temperature: float = 0.2,
        max_workers: Optional[int] = None,
        reasoning: Optional[Dict[str, str]] = None,
        post_processor_chain: Optional[PostProcessorChain] = None,
        length_reduction: Optional[Union[int, Tuple[int, int]]] = None,
        use_batch: bool = False,
        batch_size: Optional[int] = None,
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
            post_processor_chain (Optional[PostProcessorChain]): Chain of post-processors to apply
            length_reduction (Optional[Union[int, Tuple[int, int]]]): Length reduction parameter for the phase
            use_batch (bool): Whether to use batch processing for LLM calls (if supported)
            batch_size (Optional[int]): Number of items to process in each batch (if None, processes all blocks at once)
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
        self.post_processor_chain = post_processor_chain
        self.length_reduction = length_reduction
        self.use_batch = use_batch
        self.batch_size = batch_size

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
        logger.debug(f"Length reduction: {length_reduction}")

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
            if self.length_reduction is not None:
                format_params = {}
                if isinstance(self.length_reduction, int):
                    format_params["length_reduction"] = f"{self.length_reduction}%"
                elif isinstance(self.length_reduction, tuple) and len(self.length_reduction) == 2:
                    format_params["length_reduction"] = f"{self.length_reduction[0]}-{self.length_reduction[1]}%"
                else:
                    format_params["length_reduction"] = str(self.length_reduction)

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

                # Prepare batch requests
                batch_requests: List[Optional[Dict[str, Any]]] = []
                for current_block, original_block in batch:
                    current_header, current_body = self._get_header_and_body(block=current_block)
                    original_header, original_body = self._get_header_and_body(block=original_block)

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
                    batch_responses = self.model.batch_chat_completion(
                        requests=valid_requests, temperature=self.temperature, reasoning=self.reasoning, **kwargs
                    )
                else:
                    batch_responses = []

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
                        metadata = request["metadata"]
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

    def _process_block(self, current_block: str, original_block: str, **kwargs) -> str:
        """
        Process a single markdown block using the LLM model.

        This method processes a markdown block by:
        1. Extracting the header and body
        2. Formatting a user message using the prompt template
        3. Sending the message to the LLM model
        4. Applying post-processing to clean up the result

        Args:
            current_block (str): The current markdown block to process
            original_block (str): The original markdown block for reference
            **kwargs: Additional context or parameters

        Returns:
            str: The processed block content
        """
        try:
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

            # Get LLM response
            processed_body, generation_id = self.model.chat_completion(
                system_prompt=self.system_prompt,
                user_prompt=body,
            )

            # Track generation ID for cost calculation
            add_generation_id(phase_name=self.name, generation_id=generation_id)

            # Apply post-processing
            processed_body = self._apply_post_processing(
                original_block=current_body, llm_block=processed_body, **kwargs
            )

            # Reconstruct the block
            if processed_body.strip():
                return f"{current_header}\n\n{processed_body}\n\n"
            logger.debug("Empty block body, returning header only")
            return f"{current_header}\n\n"

        except Exception as e:
            logger.error(f"Error processing block: {str(e)}")
            return current_block


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
        """
        try:
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
                introduction, generation_id = self.model.chat_completion(
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                    temperature=self.temperature,
                    reasoning=self.reasoning,
                    **kwargs,
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

        except Exception as e:
            logger.error(f"Error processing block: {str(e)}")
            logger.exception("Stack trace for block processing error")
            # Return the original block to allow processing to continue
            return f"{current_header}\n\n{current_body if current_body else original_body}\n\n"


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
        """
        try:
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
                summary, generation_id = self.model.chat_completion(
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                    temperature=self.temperature,
                    reasoning=self.reasoning,
                    **kwargs,
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

        except Exception as e:
            logger.error(f"Error processing block: {str(e)}")
            logger.exception("Stack trace for block processing error")
            # Return the original block to allow processing to continue
            return f"{current_header}\n\n{current_body if current_body else original_body}\n\n"
