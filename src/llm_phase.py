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
        system_prompt_path: Path,
        book_name: str,
        author_name: str,
        model: LlmModel,
        temperature: float = 0.2,
        max_workers: int = None,
    ):
        logger.info(f"Initializing LlmPhase: {name}")
        logger.debug(f"Input file: {input_file_path}")
        logger.debug(f"Output file: {output_file_path}")
        logger.debug(f"System prompt path: {system_prompt_path}")
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
            system_prompt_path (Path): Path to the system prompt file
            book_name (str): Name of the book
            author_name (str): Name of the author
            model (LlmModel): LLM model instance
            temperature (float, optional): LLM temperature. Defaults to 0.2.
            max_workers (int, optional): Maximum worker threads for parallel processing. Defaults to None (executor default).
        """
        self.name = name
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.system_prompt_path = system_prompt_path
        self.book_name = book_name
        self.author_name = author_name
        self.model = model
        self.temperature = temperature
        self.max_workers = max_workers

        # Read file contents and prompt
        logger.info("Reading input file and system prompt")
        try:
            self.input_text = self._read_input_file()
            logger.debug(f"Read {len(self.input_text)} characters from input file")
            self.system_prompt = self._read_system_prompt()
            logger.debug("System prompt loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LlmPhase: {str(e)}")
            raise

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

    def _process_block(self, block: str, **kwargs) -> str:
        """
        Process a single Markdown block: extract header and body, run LLM on the body if present,
        and return the formatted block with header and processed content.

        Args:
            block (str): The markdown block to process
            **kwargs: Additional arguments to pass to the chat completion

        Returns:
            str: The processed markdown block
        """
        try:
            logger.debug("Processing markdown block")
            logger.trace(
                f"Block content: {block[:200]}..."
                if len(block) > 200
                else f"Block content: {block}"
            )

            lines = block.strip().split("\n", 1)
            header = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""

            logger.debug(f"Processing block with header: {header}")

            if body:
                logger.debug(f"Processing block body (length: {len(body)})")
                processed_body = self.model.chat_completion(
                    system_prompt=self.system_prompt,
                    user_prompt=body,
                    temperature=self.temperature,
                    **kwargs,
                )
                logger.debug(f"Successfully processed block: {header}")
                return f"{header}\n\n{processed_body}\n\n"
            else:
                logger.debug("Empty block body, returning header only")
                return f"{header}\n\n"

        except Exception as e:
            logger.error(f"Error processing block: {str(e)}")
            logger.exception("Stack trace for block processing error")
            # Return the original block to allow processing to continue
            return f"{header}\n\n{body if body else ''}\n\n"

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
            blocks = re.findall(pattern, self.input_text, flags=re.DOTALL)
            logger.info(f"Found {len(blocks)} markdown blocks to process")

            if not blocks:
                logger.warning("No markdown blocks found in the input text")
                self.content = ""
                return

            # Prepare for parallel processing
            func = partial(self._process_block, **kwargs)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                processed_blocks = list(
                    tqdm(
                        executor.map(func, blocks),
                        total=len(blocks),
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
