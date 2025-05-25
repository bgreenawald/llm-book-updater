from pathlib import Path

from src.llm_model import Gemini2Flash
from src.llm_phase import LlmPhase

model = Gemini2Flash()


phase = LlmPhase(
    name="modernize",
    input_file_path=Path(
        r"books\On Liberty\markdown\Mill, On Liberty\Mill, On Liberty Clean.md"
    ),
    output_file_path=Path(
        r"books\On Liberty\markdown\Mill, On Liberty\Mill, On Liberty Clean Modernized.md"
    ),
    system_prompt_path=Path("./prompts/modernize.md"),
    book_name="On Liberty",
    author_name="John Stuart Mill",
    model=model,
    temperature=0.2,
)


phase.run(reasoning_effort="medium")
