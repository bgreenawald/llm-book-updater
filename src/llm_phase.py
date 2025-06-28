import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union

from loguru import logger
from tqdm import tqdm

from src.llm_model import LlmModel
from src.logging_config import setup_logging
from src.post_processors import PostProcessorChain

# Set up logging
setup_logging("llm_phase")


class LlmPhase(ABC):
    """
    Base class for LLM processing phases. Provides common functionality for
    reading files, managing prompts, and coordinating parallel processing.
    Subclasses define specific block processing strategies.
    """

    def __init__(
        self,
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
        max_workers: int = None,
        reasoning: dict = None,
        post_processor_chain: PostProcessorChain = None,
        length_reduction: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        """
        Initialize the LLM phase.

        Args:
            name (str): Name of the processing phase
            input_file_path (Path): Path to the input Markdown file
            output_file_path (Path): Path for the processed output
            original_file_path (Path): Path for the original file (pre any
                transformations)
            system_prompt_path (Path): Path to the system prompt file
            user_prompt_path (Path): Path to the user prompt file
            book_name (str): Name of the book
            author_name (str): Name of the author
            model (LlmModel): LLM model instance
            temperature (float, optional): LLM temperature. Defaults to 0.2.
            max_workers (int, optional): Maximum worker threads for parallel
                processing. Defaults to None (executor default).
            reasoning (dict, optional): Reasoning configuration. Defaults to None.
            post_processor_chain (PostProcessorChain, optional): Post-processor
                chain for additional processing. Defaults to None.
            length_reduction (Optional[Union[int, Tuple[int, int]]], optional):
                Length reduction parameter. Defaults to None.
        """
        logger.info(f"Initializing LlmPhase: {name}")
        logger.debug(f"Input file: {input_file_path}")
        logger.debug(f"Original file: {original_file_path}")
        logger.debug(f"Output file: {output_file_path}")
        logger.debug(f"System prompt path: {system_prompt_path}")
        logger.debug(f"User prompt path: {user_prompt_path}")
        logger.debug(f"Book: {book_name} by {author_name}")
        logger.debug(f"Temperature: {temperature}, Max workers: {max_workers}")
        logger.debug(f"Length reduction: {length_reduction}")

        self.name = name
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.original_file_path = original_file_path
        self.system_prompt_path = system_prompt_path
        self.book_name = book_name
        self.author_name = author_name
        self.model = model
        self.user_prompt_path = user_prompt_path
        self.temperature = temperature
        self.max_workers = max_workers or 1
        self.reasoning = reasoning or {}
        self.post_processor_chain = post_processor_chain
        self.length_reduction = length_reduction

        if self.reasoning:
            logger.debug(f"Reasoning configuration: {self.reasoning}")

        # Read file contents and prompt
        logger.info("Reading input file and system prompt")
        try:
            self.input_text = self._read_input_file()
            logger.debug(f"Read {len(self.input_text)} characters from input file")
            self.original_text = self._read_original_file()
            logger.debug(f"Read {len(self.original_text)} characters from original file")
            self.system_prompt = self._read_system_prompt()
            logger.debug("System prompt loaded successfully")
            self.user_prompt = self._read_user_prompt()
            logger.debug("User prompt loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LlmPhase: {str(e)}")
            raise

    def __str__(self):
        """
        Returns a string representation of the LlmPhase instance.
        """
        return (
            f"{self.__class__.__name__}(name={self.name}, "
            f"input_file_path={self.input_file_path}, "
            f"output_file_path={self.output_file_path}, "
            f"original_file_path={self.original_file_path}, "
            f"system_prompt_path={self.system_prompt_path}, "
            f"book_name={self.book_name}, author_name={self.author_name}, "
            f"model={self.model}, user_prompt_path={self.user_prompt_path}, "
            f"temperature={self.temperature}, max_workers={self.max_workers}, "
            f"reasoning={self.reasoning}, "
            f"post_processor_chain={self.post_processor_chain}, "
            f"length_reduction={self.length_reduction})"
        )

    def __repr__(self):
        """
        Returns a detailed string representation of the LlmPhase instance for debugging.
        """
        return self.__str__()

    @abstractmethod
    def _process_block(self, current_block: str, original_block: str, **kwargs) -> str:
        """
        Abstract method for processing a single Markdown block.
        Subclasses must implement this to define their specific processing strategy.

        Args:
            current_block (str): The markdown block currently being processed
                (may contain content from previous phases)
            original_block (str): The completely unedited block from the original text
            **kwargs: Additional arguments to pass to the processing

        Returns:
            str: The processed markdown block
        """
        pass

    def _apply_post_processing(self, original_block: str, llm_block: str, **kwargs) -> str:
        """
        Apply post-processing to the LLM-generated block if a post-processor chain
        is configured.

        Args:
            original_block (str): The completely unedited block from the original text
            llm_block (str): The LLM-generated block
            **kwargs: Additional context for post-processors

        Returns:
            str: The post-processed block, or the original LLM block if no
                post-processors are configured
        """
        if self.post_processor_chain is None:
            return llm_block

        logger.debug(f"Applying post-processing chain with {len(self.post_processor_chain)} processors")
        try:
            processed_block = self.post_processor_chain.process(original_block, llm_block, **kwargs)
            logger.debug("Post-processing completed successfully")
            return processed_block
        except Exception as e:
            logger.error(f"Error during post-processing: {str(e)}")
            logger.exception("Post-processing error stack trace")
            # Return the original LLM block if post-processing fails
            return llm_block

    def _read_input_file(self) -> str:
        """
        Reads the content of the input file and returns it as a string.

        Returns:
            str: Content of the input file

        Raises:
            FileNotFoundError: If the input file does not exist
            IOError: If there is an error reading the file
        """
        try:
            with self.input_file_path.open("r", encoding="utf-8") as f:
                content = f.read()
                logger.debug(f"Successfully read input file: {self.input_file_path}")
                return content
        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_file_path}")
            raise
        except IOError as e:
            logger.error(f"Error reading input file {self.input_file_path}: {str(e)}")
            raise

    def _read_original_file(self) -> str:
        """
        Reads the content of the original file and returns it as a string.

        Returns:
            str: Content of the original file

        Raises:
            FileNotFoundError: If the original file does not exist
            IOError: If there is an error reading the file
        """
        try:
            with self.original_file_path.open("r", encoding="utf-8") as f:
                content = f.read()
                logger.debug(f"Successfully read original file: {self.original_file_path}")
                return content
        except FileNotFoundError:
            logger.error(f"Original file not found: {self.original_file_path}")
            raise
        except IOError as e:
            logger.error(f"Error reading original file {self.original_file_path}: {str(e)}")
            raise

    def _write_output_file(self, content: str) -> None:
        """
        Writes the content to the output file.

        Args:
            content (str): Content to write to the output file

        Raises:
            IOError: If there is an error writing to the file
        """
        try:
            self.output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_file_path.open("w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Successfully wrote output to: {self.output_file_path}")
            logger.debug(f"Wrote {len(content)} characters to output file")
        except IOError as e:
            logger.error(f"Error writing to output file {self.output_file_path}: {str(e)}")
            raise

    def _read_system_prompt(self) -> str:
        """
        Reads the content of the system prompt file and returns it as a string.
        Supports formatting with parameters like length_reduction.

        Returns:
            str: Content of the system prompt file, formatted with parameters

        Raises:
            FileNotFoundError: If the system prompt file does not exist
            IOError: If there is an error reading the file
        """
        try:
            with self.system_prompt_path.open("r", encoding="utf-8") as f:
                content = f.read()
                logger.debug(f"Read system prompt from {self.system_prompt_path}")

                # Format the prompt with available parameters
                format_params = {}
                if self.length_reduction is not None:
                    if isinstance(self.length_reduction, int):
                        format_params["length_reduction"] = f"{self.length_reduction}%"
                    else:
                        lower, upper = self.length_reduction
                        format_params["length_reduction"] = f"{lower}-{upper}%"

                if format_params:
                    try:
                        content = content.format(**format_params)
                        logger.debug(f"Formatted system prompt with parameters: {format_params}")
                    except KeyError as e:
                        logger.warning(f"System prompt contains undefined parameter: {e}")
                    except Exception as e:
                        logger.warning(f"Error formatting system prompt: {e}")

                return content

        except FileNotFoundError:
            logger.error(f"System prompt file not found: {self.system_prompt_path}")
            raise
        except IOError as e:
            logger.error(f"Error reading system prompt file {self.system_prompt_path}: {str(e)}")
            raise

    def _read_user_prompt(self) -> str:
        """
        Reads the content of the user prompt file and returns it as a string.

        Returns:
            str: Content of the user prompt file.

        Raises:
            FileNotFoundError: If the user prompt file does not exist
            IOError: If there is an error reading the file
        """
        try:
            with self.user_prompt_path.open("r", encoding="utf-8") as f:
                content = f.read()
                logger.debug(f"Read user prompt from {self.user_prompt_path}")
                return content

        except FileNotFoundError:
            logger.error(f"User prompt file not found: {self.user_prompt_path}")
            raise
        except IOError as e:
            logger.error(f"Error reading user prompt file {self.user_prompt_path}: {str(e)}")
            raise

    def _format_user_message(
        self,
        current_block: str,
        original_block: str,
        current_title: str = None,
        original_title: str = None,
    ) -> str:
        """
        Format the user message using the user prompt template.

        Args:
            current_block (str): The current block content (may contain content from previous phases)
            original_block (str): The completely unedited block content
            current_title (str, optional): The current/transformed title
            original_title (str, optional): The original title

        Returns:
            str: The formatted user message
        """
        if not current_block and not original_block:
            return ""

        # Create a context dict with all available variables
        context = {
            "transformed_passage": current_block,
            "original_passage": original_block,
            "book_name": self.book_name,
            "author_name": self.author_name,
        }

        # Add titles if provided
        if current_title is not None:
            context["new_title"] = current_title
        if original_title is not None:
            context["original_title"] = original_title

        ret = self.user_prompt.format(**context)
        return ret

    def _get_header_and_body(self, block: str):
        """
        Helper method to extract header and body from a markdown block.

        Args:
            block (str): The markdown block to parse

        Returns:
            tuple: (header, body) where header is the first line and body is the rest
        """
        lines = block.strip().split("\n", 1)
        header = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        return header, body

    def _process_markdown_blocks(self, **kwargs) -> None:
        """
        Splits the input into Markdown blocks by headers and processes each block in parallel.
        Ensures the output order matches the original order of blocks.

        Args:
            **kwargs: Additional arguments to pass to the block processing
        """
        logger.info(f"Starting to process markdown blocks with {self.max_workers} workers")

        try:
            pattern = r"(#+\s.*?)(?=\n#+\s|\Z)"
            current_blocks = re.findall(pattern, self.input_text, flags=re.DOTALL)
            original_blocks = re.findall(pattern, self.original_text, flags=re.DOTALL)
            logger.info(f"Found {len(current_blocks)} current markdown blocks to process")
            logger.info(f"Found {len(original_blocks)} original markdown blocks")

            if not current_blocks:
                logger.warning("No markdown blocks found in the input text")
                self.content = ""
                return
            elif len(current_blocks) != len(original_blocks):
                msg = f"Block length mismatch: {len(current_blocks)} != {len(original_blocks)}"
                raise ValueError(msg)

            blocks = list(zip(current_blocks, original_blocks))

            # Prepare for parallel processing
            func = partial(self._process_block, **kwargs)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                processed_blocks = list(
                    tqdm(
                        executor.map(lambda args: func(*args), blocks),
                        total=len(blocks),
                        desc=f"Processing {self.name} blocks",
                    )
                )

            logger.debug(f"Successfully processed {len(processed_blocks)} blocks")

            # Reassemble in original order
            self.content = "".join(processed_blocks)
            logger.info(f"Completed processing all blocks. Total output length: {len(self.content)} characters")

        except Exception as e:
            logger.error(f"Error during markdown block processing: {str(e)}")
            logger.exception("Stack trace for markdown block processing error")
            raise

    def run(self, **kwargs) -> None:
        """
        Run the LLM processing phase.

        Args:
            **kwargs: Additional arguments to pass to the processing methods

        Returns:
            None
        """
        logger.info(f"Starting LLM phase: {self.name}")
        try:
            self._process_markdown_blocks(**kwargs)
            self._write_output_file(self.content)
            logger.success(f"Successfully completed LLM phase: {self.name}")
        except Exception as e:
            logger.error(f"Failed to complete LLM phase {self.name}: {str(e)}")
            logger.exception("Stack trace for LLM phase error")
            raise


class StandardLlmPhase(LlmPhase):
    """
    Standard LLM phase that processes blocks by replacing the body content with LLM output.
    This maintains the original behavior of the LlmPhase class.
    """

    def _process_block(self, current_block: str, original_block: str, **kwargs) -> str:
        """
        Process a single Markdown block: extract header and body, run LLM on the body if present,
        and return the formatted block with header and processed content.

        Args:
            current_block (str): The markdown block currently being processed
                (may contain content from previous phases)
            original_block (str): The completely unedited block from the original text
            **kwargs: Additional arguments to pass to the chat completion

        Returns:
            str: The processed markdown block
        """
        try:
            current_header, current_body = self._get_header_and_body(current_block)
            original_header, original_body = self._get_header_and_body(original_block)

            body = self._format_user_message(current_body, original_body, current_header, original_header)

            if body:
                processed_body = self.model.chat_completion(
                    system_prompt=self.system_prompt,
                    user_prompt=body,
                    temperature=self.temperature,
                    reasoning=self.reasoning,
                    **kwargs,
                )
                processed_body = self._apply_post_processing(current_body, processed_body, **kwargs)
                return f"{current_header}\n\n{processed_body}\n\n"
            else:
                logger.debug("Empty block body, returning header only")
                return f"{current_header}\n\n"

        except Exception as e:
            logger.error(f"Error processing block: {str(e)}")
            logger.exception("Stack trace for block processing error")
            # Return the original block to allow processing to continue
            return f"{current_header}\n\n{current_body if current_body else ''}\n\n"


class IntroductionAnnotationPhase(LlmPhase):
    """
    LLM phase that adds introduction annotations to the beginning of each block.
    Uses the block content as input to generate an introduction, then prepends it to the block.
    """

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

            # Use the block content as the user prompt for generating the introduction
            user_prompt = self._format_user_message(current_body, original_body, current_header, original_header)

            if user_prompt:
                introduction = self.model.chat_completion(
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                    temperature=self.temperature,
                    reasoning=self.reasoning,
                    **kwargs,
                )

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

            # Use the block content as the user prompt for generating the summary
            user_prompt = self._format_user_message(current_body, original_body, current_header, original_header)

            if user_prompt:
                summary = self.model.chat_completion(
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                    temperature=self.temperature,
                    reasoning=self.reasoning,
                    **kwargs,
                )

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
