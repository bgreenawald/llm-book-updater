from pathlib import Path

from src.llm_model import Gemini2Pro
from src.llm_phase import LlmPhase

model = Gemini2Pro()


phase = LlmPhase(
    name="edit",
    input_file_path=Path(
        r"books\On Liberty\markdown\Mill, On Liberty\Mill, On Liberty Modernized.md"
    ),
    output_file_path=Path(
        r"books\On Liberty\markdown\Mill, On Liberty\Mill, On Liberty Clean Edited.md"
    ),
    system_prompt_path=Path("./prompts/edit.md"),
    book_name="On Liberty",
    author_name="John Stuart Mill",
    model=model,
    temperature=0.2,
)


phase.run(reasoning_effort="medium")
