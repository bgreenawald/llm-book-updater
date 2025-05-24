import re
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
        model: LlmModel,
        temperature: float = 0.2,
    ):
        """
        Runs an individual phase of the LLM processing. It will read the input file,
        split it into blocks based on headers, apply a processing function to the text
        content of each block, and save the processed content to the output file.

        Args:
            name (str): Name of the phase
            input_file_path (Path): Location of the input file
            output_file_path (Path): Location of the output file
            system_prompt_path (Path): Location of the system prompt
            model (LlmModel): LLM model
            temperature (float, optional): Temperature to pass to LLM. Defaults to 0.2.
        """
        self.name = name
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.temperature = temperature

        # Read the input file and system prompt
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

        Returns:
            str: Content of the system prompt file
        """
        with self.system_prompt_path.open("r", encoding="utf-8") as f:
            return f.read()

    def _process_markdown_blocks(self, **kwargs) -> None:
        """
        Processing the input content, one Markdown block at a time.

        """
        pattern = r"(#+\s.*?)(?=\n#+\s|\Z)"
        matches = re.findall(pattern, self.input_text, flags=re.DOTALL)

        processed_content = ""

        for block in tqdm(matches):
            lines = block.strip().split("\n", 1)
            header = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""

            if body:
                processed_body = self.model.chat_completion(
                    system_prompt=self.system_prompt, user_prompt=body, **kwargs
                )
                processed_content += f"{header}\n\n{processed_body}\n\n"
            else:
                processed_content += f"{header}\n\n"

        self.content = processed_content

    def run(self, **kwargs) -> None:
        self._process_markdown_blocks(**kwargs)
        self._write_output_file(self.content)
