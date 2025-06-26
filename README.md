# LLM Book Updater

A tool for processing and updating book content using Large Language Models (LLMs).

## Features

- **Markdown-First Workflow**: Easily convert PDF books to Markdown and process them.
- **Flexible Processing Pipeline**: Customize content processing with different "phases" like content modernization, editing, and adding annotations.
- **Extensible Annotation System**: Add introductions or summaries to sections without altering the original text.
- **Configurable LLM Models**: Supports various LLM providers and models.
- **Parallel Processing**: Process multiple sections concurrently for improved performance.
- **Automatic Metadata**: Keeps a JSON record of each pipeline run, tracking settings, files, and phase details.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd llm-book-updater
    ```

2.  **Install dependencies using `uv`:**
    ```bash
    uv pip install .
    ```
    *(This command uses the dependencies specified in `pyproject.toml`)*

3.  **Configure your LLM API credentials** (e.g., set `GEMINI_API_KEY` as an environment variable).

## Usage

The workflow is typically a two-step process:

### 1. Convert PDF to Markdown

Use the `marker` tool to convert your book from PDF to a clean Markdown file.

```powershell
uv run marker_single /path/to/book.pdf --output_format markdown --output_dir . --use_llm --gemini_api_key YOUR_GEMINI_API_KEY
```

### 2. Run the Processing Pipeline

Use the Python pipeline to apply transformations to the Markdown file. You can run pre-defined phases or create your own.

```python
# See examples/run_pipeline_example.py for a complete example

from src.pipeline import Pipeline
from pathlib import Path

# Configure the pipeline
pipeline = Pipeline(
    input_file=Path("input.md"),
    output_directory=Path("output/"),
    book_name="My Awesome Book",
    author_name="Author Name",
    # Other configurations...
)

# Run the desired phases
pipeline.run()
```

## Phase Types

The system supports different processing phases:

-   **`StandardLlmPhase`**: Replaces content with LLM-generated output. Useful for tasks like modernizing or editing text.
-   **`IntroductionAnnotationPhase`**: Adds an LLM-generated introduction to the beginning of each section.
-   **`SummaryAnnotationPhase`**: Adds an LLM-generated summary to the end of each section.

Phases can be chained together to create complex processing workflows.

## Configuration

### Phase Configuration

All phases can be configured with the following:

-   `name`: Name of the processing phase.
-   `input_file_path`, `output_file_path`, `original_file_path`: Paths for input, output, and original files.
-   `system_prompt_path`, `user_prompt_path`: Paths to prompt files.
-   `book_name`, `author_name`: Metadata for prompts.
-   `model`: The `LlmModel` instance to use.
-   `temperature`, `max_workers`, etc.

### Prompt Templates

System and user prompts can be customized with template variables like `{book_name}`, `{author_name}`, `{transformed_passage}`, and `{original_passage}`.

## Examples

See the `examples/` directory for complete working examples:

-   `annotation_example.py`: Demonstrates all three phase types.
-   `run_pipeline_example.py`: Shows how to run the full pipeline.
-   `metadata_example.py`: Example of how metadata is saved.
-   `retry_example.py`: Example with retry logic.

## License

[Add your license information here]
