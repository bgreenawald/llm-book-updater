# LLM Book Updater

A tool for processing and updating book content using Large Language Models (LLMs).

## Features

- **Markdown-First Workflow**: Easily convert PDF books to Markdown and process them.
- **Flexible Processing Pipeline**: Customize content processing with different "phases" like content modernization, editing, and adding annotations.
- **Extensible Annotation System**: Add introductions or summaries to sections without altering the original text.
- **Configurable LLM Models**: Supports various LLM providers and models.
- **Parallel Processing**: Process multiple sections concurrently for improved performance.
- **Automatic Metadata**: Keeps a JSON record of each pipeline run, tracking settings, files, and phase details.
- **Post-Processing Pipeline**: Configurable post-processing chain with detailed logging and metadata tracking.
- **Cost Tracking**: Monitor and log OpenRouter API usage costs per phase and total run costs (optional add-on).

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

3.  **Configure your LLM API credentials** (e.g., set `OPENROUTER_API_KEY` as an environment variable).

## System Requirements

For PDF generation functionality, you'll need to install additional system dependencies:

### LaTeX Distribution
A LaTeX distribution is required for some PDF processing features. Install one of the following:

- **TeX Live** (recommended for Linux/macOS):
  ```bash
  # Ubuntu/Debian
  sudo apt-get install texlive-full
  
  # macOS (using Homebrew)
  brew install --cask mactex
  
  # Or install basic TeX Live
  brew install texlive
  ```

- **MiKTeX** (recommended for Windows):
  - Download and install from [MiKTeX website](https://miktex.org/download)

### WeasyPrint
WeasyPrint is used for optimal CSS rendering in PDF generation:

```bash
# Install WeasyPrint and its dependencies
pip install weasyprint

# On Ubuntu/Debian, you may also need:
sudo apt-get install build-essential python3-dev python3-pip python3-setuptools python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info

# On macOS:
brew install cairo pango gdk-pixbuf libffi
```

*Note: WeasyPrint provides the best CSS rendering and formatting support for PDF generation.*

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

## Cost Tracking

The system includes an optional cost tracking feature for OpenRouter API usage. This allows you to monitor and log costs for each phase and the total run without modifying existing code.

### Quick Start with Cost Tracking

1. Set your OpenRouter API key:
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```

2. Use the cost tracking wrapper in your code:
   ```python
   from src.cost_tracking_wrapper import add_generation_id, calculate_and_log_costs
   
   # After each API call, add the generation ID
   processed_body, generation_id = self.model.chat_completion(...)
   add_generation_id(phase_name=self.name, generation_id=generation_id)
   
   # At the end of your pipeline, calculate and log costs
   phase_names = ["modernize", "edit", "final"]
   run_costs = calculate_and_log_costs(phase_names)
   ```

3. Run the example:
   ```bash
   python examples/cost_tracking_example.py
   ```

## Examples

See the `examples/` directory for complete working examples:

-   `annotation_example.py`: Demonstrates all three phase types.
-   `run_pipeline_example.py`: Shows how to run the full pipeline.
-   `metadata_example.py`: Example of how metadata is saved.
-   `retry_example.py`: Example with retry logic.
-   `default_postprocessors_example.py`: Demonstrates default post-processor configurations.
