# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core library code for pipeline, models, phases, and utilities.
- `cli/`: command-line entry points (`python -m cli ...`) and subcommands.
- `books/`: book-specific build/run modules and assets.
- `examples/`: runnable usage examples for pipeline features.
- `tests/`: pytest suite; `conftest.py` provides shared fixtures.
- `build/`, `logs/`, `htmlcov/`: generated artifacts (do not edit by hand).

## Build, Test, and Development Commands
- `uv pip install .`: install runtime dependencies from `pyproject.toml`.
- `uv pip install ".[dev]"`: install dev tools (ruff, pytest, mypy, bandit, pre-commit).
- `python -m cli build <book_name> <version>`: build EPUB/PDF for a book.
- `python -m cli run <book_name>`: run the processing pipeline for a book.
- `python -m cli --help`: list available CLI commands and options.
- `pytest`: run the full test suite in `tests/`.
- `ruff check .` / `ruff format .`: lint and format the codebase.

## Coding Style & Naming Conventions
- Python 3.12; 4-space indentation; double quotes; max line length 120 (`ruff`).
- Use `snake_case` for functions/variables and `PascalCase` for classes.
- Keep CLI behavior in `cli/` and reusable logic in `src/`.

## Testing Guidelines
- Framework: `pytest` with discovery rules in `pyproject.toml`.
- File naming: `tests/test_*.py`; test functions `test_*`, classes `Test*`.
- Add focused unit tests for new phases or processors; prefer fast, deterministic tests.

## Commit & Pull Request Guidelines
- Recent history uses concise, imperative messages (e.g., "Handle failed case").
- If applicable, reference issues/PRs in the description and explain behavior changes.
- PRs should include: a clear summary, test commands run, and any sample outputs.

## Configuration & Secrets
- LLM provider keys are required; set `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, or
  `GEMINI_API_KEY` in your shell or `.env` (loaded via `python-dotenv`).
- Avoid committing generated artifacts or credentials.
