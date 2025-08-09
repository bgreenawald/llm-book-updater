
# Project Overview

This project, "LLM Book Updater," is a Python-based tool for processing and updating book content using Large Language Models (LLMs). It provides a flexible pipeline for transforming Markdown files, allowing for content modernization, editing, and adding annotations. The system is designed to be extensible, supporting various LLM models and a configurable post-processing chain.

**Key Technologies:**

*   **Programming Language:** Python 3.12
*   **Core Libraries:**
    *   `loguru` for logging
    *   `marker-pdf` for PDF to Markdown conversion
    *   `openai` for interacting with LLM APIs
    *   `pypandoc` for document format conversion
    *   `click` for the command-line interface
*   **Dependency Management:** `uv` and `pip` with `pyproject.toml`
*   **Code Quality:** `ruff` for linting and formatting, `pre-commit` for git hooks.

**Architecture:**

The application is structured around a pipeline concept. The `src/pipeline.py` file defines the `Pipeline` class, which orchestrates the execution of different processing "phases." Each phase (e.g., "modernize," "edit") is configured via the `src/config.py` file and can have its own LLM model, prompts, and post-processors. The `cli` directory provides a command-line interface for running the pipeline and building the final book outputs. The `books` directory contains the source materials and build scripts for different books.

# Building and Running

**Installation:**

1.  Clone the repository.
2.  Install dependencies using `uv`:
    ```bash
    uv pip install .
    ```

**Running the Pipeline:**

The primary way to use the tool is through the command-line interface.

*   **Run pipeline processing for a book:**
    ```bash
    python -m cli run <book_name>
    ```
    For example:
    ```bash
    python -m cli run on_liberty
    ```

*   **Build a book from processed Markdown to EPUB/PDF:**
    ```bash
    python -m cli build <book_name> <version>
    ```
    For example:
    ```bash
    python -m cli build the_federalist_papers v1.0.0
    ```

**Testing:**

While no explicit top-level test command is found, the project uses `pytest`. To run tests, you would likely use:

```bash
# TODO: Confirm the exact test command.
pytest
```

# Development Conventions

*   **Code Style:** The project uses `ruff` for formatting, configured in `pyproject.toml`. It follows a style similar to Black, with a line length of 120 characters and double quotes for strings.
*   **Dependency Management:** Dependencies are managed in `pyproject.toml` and installed using `uv`.
*   **Modularity:** The project is organized into distinct modules: `cli` for the command-line interface, `src` for the core logic, `books` for book-specific content, and `prompts` for the LLM prompts.
*   **Configuration:** The pipeline is configured through Python data classes in `src/config.py`, allowing for type-safe and structured configuration.
*   **Extensibility:** New books can be added by creating a new directory in the `books` folder with a corresponding `build.py` script. New processing phases can be added by extending the `PhaseType` enum in `src/config.py` and creating corresponding prompt files.
