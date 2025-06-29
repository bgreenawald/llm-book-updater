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

For detailed information about cost tracking, see [COST_TRACKING.md](COST_TRACKING.md).

## Configuration

### Phase Configuration

All phases can be configured with the following:

-   `name`: Name of the processing phase.
-   `input_file_path`, `output_file_path`, `original_file_path`: Paths for input, output, and original files.
-   `system_prompt_path`, `user_prompt_path`: Paths to prompt files.
-   `book_name`, `author_name`: Metadata for prompts.
-   `model`: The `LlmModel` instance to use.
-   `temperature`, `max_workers`, etc.

### Run Configuration

The `RunConfig` class supports the following parameters:

-   `book_name`, `author_name`: Metadata for the book being processed.
-   `input_file`, `output_dir`, `original_file`: Paths for the run.
-   `phases`: List of `PhaseConfig` objects defining the processing pipeline.
-   `length_reduction`: Length reduction parameter for the entire run (optional).

### Length Reduction Parameter

The pipeline supports a `length_reduction` parameter that can be specified at the run level. This parameter controls how much the content should be reduced in length during the editing phase.

### Usage

The `length_reduction` parameter can be:
- A single integer (e.g., `40` for 40% reduction)
- A tuple of bounds (e.g., `(35, 50)` for 35-50% reduction)
- `None` to use the default of `(35, 50)` (35-50% reduction)

### Examples

```python
from src.config import RunConfig, PhaseConfig, PhaseType
from src.pipeline import Pipeline

# Single percentage reduction
config = RunConfig(
    book_name="My Book",
    author_name="Author Name", 
    input_file=Path("input.txt"),
    output_dir=Path("output"),
    original_file=Path("original.txt"),
    phases=[PhaseConfig(phase_type=PhaseType.EDIT)],
    length_reduction=40  # 40% reduction
)

# Range reduction (default behavior)
config = RunConfig(
    book_name="My Book",
    author_name="Author Name",
    input_file=Path("input.txt"), 
    output_dir=Path("output"),
    original_file=Path("original.txt"),
    phases=[PhaseConfig(phase_type=PhaseType.EDIT)],
    length_reduction=(35, 50)  # 35-50% reduction
)

# Use default (35-50% reduction)
config = RunConfig(
    book_name="My Book",
    author_name="Author Name",
    input_file=Path("input.txt"),
    output_dir=Path("output"), 
    original_file=Path("original.txt"),
    phases=[PhaseConfig(phase_type=PhaseType.EDIT)]
    # length_reduction defaults to (35, 50)
)
```

### Post-Processor Types

The system provides a `PostProcessorType` enum for type-safe post-processor configuration:

```python
from src.config import PostProcessorType

# Available post-processor types:
PostProcessorType.ENSURE_BLANK_LINE           # Ensures proper spacing
PostProcessorType.REMOVE_TRAILING_WHITESPACE  # Cleans trailing whitespace
PostProcessorType.REMOVE_XML_TAGS             # Removes XML tags (except <br>)
PostProcessorType.NO_NEW_HEADERS              # Prevents new markdown headers
PostProcessorType.REVERT_REMOVED_BLOCK_LINES  # Restores removed block comments
PostProcessorType.ORDER_QUOTE_ANNOTATION      # Reorders quotes before annotations
```

### Default Post-Processor Configuration

Each phase type comes with a default set of post-processors that are automatically applied when no explicit `post_processors` are specified in the configuration:

- **MODERNIZE, EDIT, FINAL, INTRODUCTION**: Basic formatting and cleanup
  - `no_new_headers`: Prevents addition of new markdown headers
  - `remove_trailing_whitespace`: Cleans up trailing whitespace
  - `remove_xml_tags`: Removes XML tags (except `<br>`)
  - `ensure_blank_line`: Ensures proper spacing between elements

- **SUMMARY**: Includes block restoration in addition to basic formatting
  - `revert_removed_block_lines`: Restores removed block comment lines
  - `no_new_headers`: Prevents addition of new markdown headers
  - `remove_trailing_whitespace`: Cleans up trailing whitespace
  - `remove_xml_tags`: Removes XML tags (except `<br>`)
  - `ensure_blank_line`: Ensures proper spacing between elements

- **ANNOTATE**: Includes quote/annotation ordering in addition to block restoration
  - `revert_removed_block_lines`: Restores removed block comment lines
  - `order_quote_annotation`: Reorders quotes before annotations
  - `no_new_headers`: Prevents addition of new markdown headers
  - `remove_trailing_whitespace`: Cleans up trailing whitespace
  - `remove_xml_tags`: Removes XML tags (except `<br>`)
  - `ensure_blank_line`: Ensures proper spacing between elements

You can override these defaults by explicitly specifying the `post_processors` parameter in your `PhaseConfig`.

### Specifying Post-Processors

You can specify post-processors in multiple ways:

```python
from src.config import PhaseConfig, PhaseType, PostProcessorType
from src.post_processors import RemoveTrailingWhitespaceProcessor

# 1. Using string names (legacy approach)
config1 = PhaseConfig(
    phase_type=PhaseType.MODERNIZE,
    post_processors=["remove_xml_tags", "ensure_blank_line"],
)

# 2. Using PostProcessorType enum (type-safe approach)
config2 = PhaseConfig(
    phase_type=PhaseType.MODERNIZE,
    post_processors=[
        PostProcessorType.REMOVE_XML_TAGS,
        PostProcessorType.ENSURE_BLANK_LINE,
    ],
)

# 3. Mixing different approaches
config3 = PhaseConfig(
    phase_type=PhaseType.ANNOTATE,
    post_processors=[
        "revert_removed_block_lines",  # String
        PostProcessorType.ORDER_QUOTE_ANNOTATION,  # Enum
        RemoveTrailingWhitespaceProcessor(),  # Instance
    ],
)
```

### Post-Processing Logging

The system provides detailed logging for post-processing operations:

- **Phase Initialization**: When a phase is initialized, it logs the post-processor pipeline configuration
- **Processing Execution**: During processing, each post-processor application is logged with progress information
- **Chain Completion**: When the post-processing chain completes, it logs the final result

Example log output:
```
2024-01-01 12:00:00 | INFO | Post-processing pipeline for modernize: ['no_new_headers', 'remove_trailing_whitespace', 'remove_xml_tags', 'ensure_blank_line']
2024-01-01 12:00:00 | INFO | Post-processor count: 4
2024-01-01 12:00:00 | DEBUG | Starting post-processing chain with 4 processors
2024-01-01 12:00:00 | DEBUG | Applying post-processor 1/4: no_new_headers
2024-01-01 12:00:00 | DEBUG | Post-processor no_new_headers completed successfully
...
2024-01-01 12:00:00 | DEBUG | Post-processing chain completed. Final block length: 1234 characters
```

### Metadata and Logging

The pipeline automatically saves comprehensive metadata about each run, including post-processing information and system prompt details:

```json
{
  "metadata_version": "1.0.0",
  "run_timestamp": "2024-01-01T12:00:00",
  "book_name": "Example Book",
  "author_name": "Example Author",
  "input_file": "input.md",
  "original_file": "original.md",
  "output_directory": "output/",
  "length_reduction": [35, 50],
  "phases": [
    {
      "phase_name": "modernize",
      "phase_index": 0,
      "phase_type": "MODERNIZE",
      "enabled": true,
      "model_type": "gemini-flash",
      "temperature": 0.2,
      "max_workers": null,
      "input_file": "input.md",
      "output_file": "01-input Modernize_1.md",
      "system_prompt_path": "./prompts/modernize_system.md",
      "user_prompt_path": "./prompts/modernize_user.md",
      "fully_rendered_system_prompt": "You are an expert editor...",
      "length_reduction_parameter": [35, 50],
      "post_processors": ["no_new_headers", "remove_trailing_whitespace", "remove_xml_tags", "ensure_blank_line"],
      "post_processor_count": 4,
      "completed": true,
      "output_exists": true,
      "book_name": "Example Book",
      "author_name": "Example Author"
    }
  ]
}
```

The consolidated metadata includes:
- **Metadata Version**: Version number for parser compatibility
- **Run Information**: Timestamp, book details, file paths, and configuration
- **Phase Details**: Complete configuration for each phase including model settings
- **System Prompts**: Fully rendered system prompts with all template variables resolved
- **Post-processing**: List of post-processor names and counts used in each phase
- **Execution Status**: Whether each phase completed successfully, was disabled, or failed
- **File Information**: Input/output file paths and existence status

The metadata is saved to a single file named `pipeline_metadata_YYYYMMDD_HHMMSS.json` in the output directory.

### Prompt Templates

System and user prompts can be customized with template variables like `{book_name}`, `{author_name}`, `{current_body}`, and `{original_body}`.

## Examples

See the `examples/` directory for complete working examples:

-   `annotation_example.py`: Demonstrates all three phase types.
-   `run_pipeline_example.py`: Shows how to run the full pipeline.
-   `metadata_example.py`: Example of how metadata is saved.
-   `retry_example.py`: Example with retry logic.
-   `default_postprocessors_example.py`: Demonstrates default post-processor configurations.

## License

[Add your license information here]
