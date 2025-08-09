# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core library (pipeline, phases, post-processors, config, cost tracking).
- `cli/`: Click-based CLI (`python -m cli build|run`).
- `books/`: Book-specific builders and assets; see `books/README.md`.
- `build/`: Versioned build outputs per book (e.g., `build/the_federalist_papers/v1.0.0/`).
- `examples/`: End-to-end usage samples.
- `tests/`: Pytest suite (`tests/test_*.py`).
- `.github/`: CI workflows; runs pre-commit and tests via `uv`.

## Build, Test, and Development Commands
- Setup env: `uv sync --dev` (Python 3.12+).
- Run tests: `uv run python -m pytest -v` (add `-k <pattern>` to filter).
- Lint & format: `uv run ruff check --fix . && uv run ruff format .`.
- Type check: `uv run mypy --ignore-missing-imports .`.
- Security scan: `uv run bandit -ll -x tests/ -r src cli books`.
- CLI help: `python -m cli --help`.
- Build a book: `python -m cli build the_federalist_papers v1.0.0`.
- Run pipeline: `python -m cli run on_liberty`.

## Coding Style & Naming Conventions
- Python style enforced by Ruff (line length 120, double quotes, spaces).
- Prefer type hints; keep functions small and pure where practical.
- Modules: snake_case; classes: PascalCase; functions/vars: snake_case.
- Books: directory ids in snake_case (e.g., `the_wealth_of_nations`).

## Testing Guidelines
- Framework: Pytest; tests live in `tests/` as `test_*.py`.
- Write fast, isolated unit tests for processors and config; mock network/LLM calls (see `tests/conftest.py`).
- Run with coverage when changing core logic: `uv run pytest --cov=src --cov-report=term-missing`.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (e.g., "CLI refactor", "Bandit fix"); include focused diffs.
- Reference issues in body (e.g., "Closes #123").
- PRs: clear description, reasoning, test plan (commands run), and screenshots or sample CLI output where useful.
- CI must pass (pre-commit, lint, tests). Include any migration notes in PR description.

## Security & Configuration Tips
- Do not commit API keys. Use env vars (e.g., `OPENROUTER_API_KEY`).
- Large artifacts belong in `build/` and are not tracked; keep inputs/outputs reproducible.
- Use `bandit` locally for security checks and follow CI findings.
