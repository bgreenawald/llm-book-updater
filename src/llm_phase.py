import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.llm_model import LlmModel
from src.logging_config import setup_logging

# Set up logging
logger = setup_logging("llm_phase")


class LlmPhase:
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
    ):
        logger.info(f"Initializing LlmPhase: {name}")
        logger.debug(f"Input file: {input_file_path}")
        logger.debug(f"Original file: {original_file_path}")
        logger.debug(f"Output file: {output_file_path}")
        logger.debug(f"System prompt path: {system_prompt_path}")
        logger.debug(f"User prompt path: {user_prompt_path}")
        logger.debug(f"Book: {book_name} by {author_name}")
        logger.debug(f"Temperature: {temperature}, Max workers: {max_workers}")
        """
        Runs an individual phase of the LLM processing in parallel. Reads the input file,
        splits it into Markdown blocks based on headers, processes blocks concurrently,
        and writes the ordered results to the output file.

        Args:
            name (str): Name of the processing phase
            input_file_path (Path): Path to the input Markdown file
            output_file_path (Path): Path for the processed output
            original_file_path (Path): Path for the original file (pre any transformations)
            system_prompt_path (Path): Path to the system prompt file
            user_prompt_path (Path, optional): Path to the user prompt file. Defaults to None
            book_name (str): Name of the book
            author_name (str): Name of the author
            model (LlmModel): LLM model instance
            temperature (float, optional): LLM temperature. Defaults to 0.2.
            max_workers (int, optional): Maximum worker threads for parallel processing. Defaults to None (executor default).
        """
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
        self.max_workers = max_workers
        self.reasoning = reasoning or {}

        if self.reasoning:
            logger.debug(f"Reasoning configuration: {self.reasoning}")

        # Read file contents and prompt
        logger.info("Reading input file and system prompt")
        try:
            self.input_text = self._read_input_file()
            logger.debug(f"Read {len(self.input_text)} characters from input file")
            self.original_text = self._read_original_file()
            logger.debug(
                f"Read {len(self.original_text)} characters from original file"
            )
            self.system_prompt = self._read_system_prompt()
            logger.debug("System prompt loaded successfully")
            self.user_prompt = self._read_user_prompt()
            logger.debug("User prompt loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LlmPhase: {str(e)}")
            raise

    def __str__(self):
        return f"LlmPhase(name={self.name}, input_file_path={self.input_file_path}, output_file_path={self.output_file_path}, original_file_path={self.original_file_path}, system_prompt_path={self.system_prompt_path}, book_name={self.book_name}, author_name={self.author_name}, model={self.model}, user_prompt_path={self.user_prompt_path}, temperature={self.temperature}, max_workers={self.max_workers}, reasoning={self.reasoning})"

    def __repr__(self):
        return f"LlmPhase(name={self.name}, input_file_path={self.input_file_path}, output_file_path={self.output_file_path}, original_file_path={self.original_file_path}, system_prompt_path={self.system_prompt_path}, book_name={self.book_name}, author_name={self.author_name}, model={self.model}, user_prompt_path={self.user_prompt_path}, temperature={self.temperature}, max_workers={self.max_workers}, reasoning={self.reasoning})"

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
                logger.debug(
                    f"Successfully read original file: {self.original_file_path}"
                )
                return content
        except FileNotFoundError:
            logger.error(f"Original file not found: {self.original_file_path}")
            raise
        except IOError as e:
            logger.error(
                f"Error reading original file {self.original_file_path}: {str(e)}"
            )
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
            logger.error(
                f"Error writing to output file {self.output_file_path}: {str(e)}"
            )
            raise

    def _read_system_prompt(self) -> str:
        """
        Reads the content of the system prompt file and returns it as a string.
        Adds in the author and book name.

        Returns:
            str: Content of the system prompt file

        Raises:
            FileNotFoundError: If the system prompt file does not exist
            IOError: If there is an error reading the file
            KeyError: If the template contains invalid placeholders
        """
        try:
            with self.system_prompt_path.open("r", encoding="utf-8") as f:
                content = f.read()
                logger.debug(f"Read system prompt from {self.system_prompt_path}")

                formatted_content = content.format(
                    author_name=self.author_name,
                    book_name=self.book_name,
                )
                logger.trace(f"Formatted system prompt: {formatted_content[:200]}...")

                return formatted_content

        except FileNotFoundError:
            logger.error(f"System prompt file not found: {self.system_prompt_path}")
            raise
        except IOError as e:
            logger.error(
                f"Error reading system prompt file {self.system_prompt_path}: {str(e)}"
            )
            raise
        except KeyError as e:
            logger.error(f"Invalid placeholder in system prompt template: {str(e)}")
            raise

    def _read_user_prompt(self) -> str:
        """
        Reads the content of the user prompt file (if it exists) and returns it as a string.

        Returns:
            str: Content of the system prompt file, if it exists, or an empty string.

        Raises:
            FileNotFoundError: If the system prompt file is defined but does not exist
            IOError: If there is an error reading the file
        """
        if not self.user_prompt_path:
            return ""
        try:
            with self.user_prompt_path.open("r", encoding="utf-8") as f:
                content = f.read()
                logger.debug(f"Read user prompt from {self.user_prompt_path}")
                return content

        except FileNotFoundError:
            logger.error(f"User prompt file not found: {self.user_prompt_path}")
            raise
        except IOError as e:
            logger.error(
                f"Error reading user prompt file {self.user_prompt_path}: {str(e)}"
            )
            raise

    def _format_user_message(self, new_block: str, original_block: str) -> str:
        if not new_block and not original_block:
            return ""

        if self.user_prompt:
            ret = self.user_prompt.format(
                transformed_passage=new_block,
                original_passage=original_block,
            )
            return ret
        logger.debug("No user prompt defined")
        return new_block

    def _process_block(self, new_block: str, original_block: str, **kwargs) -> str:
        """
        Process a single Markdown block: extract header and body, run LLM on the body if present,
        and return the formatted block with header and processed content.

        Args:
            new_block (str): The markdown block to process
            original_block (str): The original counterpart to the new block
            **kwargs: Additional arguments to pass to the chat completion

        Returns:
            str: The processed markdown block
        """
        try:

            def _get_header_and_body(block: str):
                lines = block.strip().split("\n", 1)
                header = lines[0].strip()
                body = lines[1].strip() if len(lines) > 1 else ""
                return header, body

            new_header, new_body = _get_header_and_body(new_block)
            original_header, original_body = _get_header_and_body(original_block)

            body = self._format_user_message(new_body, original_body)

            if body:
                processed_body = self.model.chat_completion(
                    system_prompt=self.system_prompt,
                    user_prompt=body,
                    temperature=self.temperature,
                    reasoning=self.reasoning,
                    **kwargs,
                )
                return f"{new_header}\n\n{processed_body}\n\n"
            else:
                logger.debug("Empty block body, returning header only")
                return f"{new_header}\n\n"

        except Exception as e:
            logger.error(f"Error processing block: {str(e)}")
            logger.exception("Stack trace for block processing error")
            # Return the original block to allow processing to continue
            return f"{new_header}\n\n{new_body if new_body else ''}\n\n"

    def _process_markdown_blocks(self, **kwargs) -> None:
        """
        Splits the input into Markdown blocks by headers and processes each block in parallel.
        Ensures the output order matches the original order of blocks.

        Args:
            **kwargs: Additional arguments to pass to the block processing
        """
        logger.info(
            f"Starting to process markdown blocks with {self.max_workers} workers"
        )

        try:
            pattern = r"(#+\s.*?)(?=\n#+\s|\Z)"
            new_blocks = re.findall(pattern, self.input_text, flags=re.DOTALL)
            original_blocks = re.findall(pattern, self.original_text, flags=re.DOTALL)
            logger.info(f"Found {len(new_blocks)} new markdown blocks to process")
            logger.info(f"Found {len(original_blocks)} markdown blocks")

            if not new_blocks:
                logger.warning("No markdown blocks found in the input text")
                self.content = ""
                return
            elif len(new_blocks) != len(original_blocks):
                msg = f"Block length mismatch: {len(new_blocks)} != {len(original_blocks)}"
                raise ValueError(msg)

            blocks = zip(new_blocks, original_blocks)

            # Prepare for parallel processing
            func = partial(self._process_block, **kwargs)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                processed_blocks = list(
                    tqdm(
                        executor.map(lambda args: func(*args), blocks),
                        total=len(list(blocks)),
                        desc=f"Processing {self.name} blocks",
                    )
                )

            logger.debug(f"Successfully processed {len(processed_blocks)} blocks")

            # Reassemble in original order
            self.content = "".join(processed_blocks)
            logger.info(
                f"Completed processing all blocks. Total output length: {len(self.content)} characters"
            )

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
