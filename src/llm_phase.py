import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from tqdm import tqdm

from src.llm_model import LlmModel


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
        self.input_text = self._read_input_file()
        self.system_prompt = self._read_system_prompt()

    def _read_input_file(self) -> str:
        """
        Reads the content of the input file and returns it as a string.

        Returns:
            str: Content of the input file
        """
        with self.input_file_path.open("r", encoding="utf-8") as f:
            return f.read()

    def _write_output_file(self, content: str) -> None:
        """
        Writes the content to the output file.

        Args:
            content (str): Content to write to the output file
        """
        with self.output_file_path.open("w", encoding="utf-8") as f:
            f.write(content)

    def _read_system_prompt(self) -> str:
        """
        Reads the content of the system prompt file and returns it as a string.
        Adds in the author and book name.

        Returns:
            str: Content of the system prompt file
        """
        with self.system_prompt_path.open("r", encoding="utf-8") as f:
            content = f.read()
            return content.format(
                author_name=self.author_name,
                book_name=self.book_name,
            )

    def _process_block(self, block: str, **kwargs) -> str:
        """
        Process a single Markdown block: extract header and body, run LLM on the body if present,
        and return the formatted block with header and processed content.
        """
        lines = block.strip().split("\n", 1)
        header = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        if body:
            processed_body = self.model.chat_completion(
                system_prompt=self.system_prompt,
                user_prompt=body,
                temperature=self.temperature,
                **kwargs,
            )
            return f"{header}\n\n{processed_body}\n\n"
        else:
            return f"{header}\n\n"

    def _process_markdown_blocks(self, **kwargs) -> None:
        """
        Splits the input into Markdown blocks by headers and processes each block in parallel.
        Ensures the output order matches the original order of blocks.
        """
        pattern = r"(#+\s.*?)(?=\n#+\s|\Z)"
        blocks = re.findall(pattern, self.input_text, flags=re.DOTALL)

        # Prepare for parallel processing
        func = partial(self._process_block, **kwargs)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            processed_blocks = list(tqdm(executor.map(func, blocks), total=len(blocks)))

        # Reassemble in original order
        self.content = "".join(processed_blocks)

    def run(self, **kwargs) -> None:
        self._process_markdown_blocks(**kwargs)
        self._write_output_file(self.content)
