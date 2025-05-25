from pathlib import Path

from src.llm_model import Gemini2Flash
from src.llm_phase import LlmPhase

model = Gemini2Flash()


phase = LlmPhase(
    name="annotate",
    input_file_path=Path(
        r"books\On Liberty\markdown\Mill, On Liberty\Mill, On Liberty Edited.md"
    ),
    output_file_path=Path(
        r"books\On Liberty\markdown\Mill, On Liberty\Mill, On Liberty Clean Annotated.md"
    ),
    system_prompt_path=Path("./prompts/annotate.md"),
    book_name="On Liberty",
    author_name="John Stuart Mill",
    model=model,
    temperature=0.2,
)


phase.run(reasoning_effort="medium")
