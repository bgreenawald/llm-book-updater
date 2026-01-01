# LLM Book Updater

A monorepo for processing, updating, and generating book content using Large Language Models (LLMs).

## Packages

This repository contains three packages:

- **[llm-core](packages/llm-core/)**: Shared LLM provider infrastructure, cost tracking, and utilities
- **[book-updater](packages/book-updater/)**: Phase-based text transformation pipeline for updating existing texts
- **[book-writer](packages/book-writer/)**: Async book generation from outlines

## Features

### Book Updater
- **Markdown-First Workflow**: Easily convert PDF books to Markdown and process them.
- **Flexible Processing Pipeline**: Customize content processing with different "phases" like content modernization, editing, and adding annotations.
- **Extensible Annotation System**: Add introductions or summaries to sections without altering the original text.
- **Parallel Processing**: Process multiple sections concurrently for improved performance.
- **Automatic Metadata**: Keeps a JSON record of each pipeline run, tracking settings, files, and phase details.
- **Post-Processing Pipeline**: Configurable post-processing chain with detailed logging and metadata tracking.

### Book Writer
- **Async Generation**: Generate complete books from outlines with parallel chapter/section processing.
- **State Management**: Resume interrupted generation with persistent state tracking.

### Shared (LLM Core)
- **Configurable LLM Models**: Supports various LLM providers (OpenRouter, OpenAI, Gemini, Claude).
- **Cost Tracking**: Monitor and log API usage costs per phase and total run costs.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd llm-book-updater
    ```

2.  **Install dependencies using `uv`:**
    ```bash
    uv sync
    ```
    *(This installs all workspace packages in editable mode)*

3.  **Configure your LLM API credentials** (see [LLM Provider Configuration](#llm-provider-configuration) below).

## Project Structure

```
llm-book-updater/
├── packages/
│   ├── llm-core/                    # Shared infrastructure
│   │   └── src/llm_core/
│   │       ├── providers/           # Multi-provider LLM clients
│   │       ├── cost/                # Cost tracking
│   │       ├── tokens/              # Token counting
│   │       └── config/              # Settings
│   │
│   ├── book-updater/                # Text transformation
│   │   └── src/book_updater/
│   │       ├── phases/              # Phase implementations
│   │       ├── processing/          # Post-processors
│   │       └── pipeline.py
│   │
│   └── book-writer/                 # Book generation
│       └── src/book_writer/
│           ├── generator.py         # Async orchestration
│           ├── state.py             # State management
│           └── models.py
├── books/                           # Book configurations
├── cli/                             # CLI entry points
├── examples/                        # Usage examples
├── tests/                           # Test suite
└── pyproject.toml                   # UV workspace root
```

## System Requirements

For PDF generation functionality, you'll need to install additional system dependencies:

### LaTeX Distribution
A LaTeX distribution is required for some PDF processing features. See the [LaTeX installation guide](https://www.latex-project.org/get/) for your platform.

### Calibre
Calibre is used for EPUB to PDF conversion with perfect formatting and working table of contents. See the [Calibre installation guide](https://calibre-ebook.com/download) for your platform.

*Note: Calibre's ebook-convert provides the best EPUB to PDF conversion with working table of contents, proper formatting, and cover support.*

## Usage

### Book Updater (update existing books)

The workflow is typically a two-step process:

1. **Convert PDF to Markdown**

Use the `marker` tool to convert your book from PDF to a clean Markdown file.

```powershell
uv run marker_single /path/to/book.pdf --output_format markdown --output_dir . --use_llm --gemini_api_key YOUR_GEMINI_API_KEY
```

2. **Run the Processing Pipeline**

Use the Python pipeline to apply transformations to the Markdown file. You can run pre-defined phases or create your own.

```python
# See examples/run_pipeline_example.py for a complete example

from book_updater import Pipeline
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

You can also run pipeline processing through the CLI for books defined in `books/`:

```bash
python -m cli run <book_name>
```

### Book Writer (generate a new book from an outline)

The book writer is an outline-first workflow. It reads a rubric file, generates per-section drafts, and tracks progress
so you can resume later.

1. **Initialize a new project**

```bash
python -m book_writer init books/my_new_book --title "My New Book"
```

This creates `config.yaml`, `rubric.md`, and an `output/` folder.

2. **Edit `rubric.md` with your outline**

The parser looks for H1 chapter headers and H2 section headers. A minimal structure looks like this:

```markdown
# My New Book

## Chapter Goals
- What this chapter should accomplish

# Chapter 1: Opening Ideas

## 1.1 First Section
### Guidance for the model
- Bullet points, prompts, or notes

## 1.2 Second Section
### Guidance for the model
- More detail
```

Supported headers include `# Preface`, `# Chapter N: Title`, `# Appendix A: Title`, and `# Final Notes`.

3. **Generate content**

The book writer currently uses OpenRouter for generation, so make sure `OPENROUTER_API_KEY` is set in your environment.

```bash
python -m book_writer generate books/my_new_book
```

Helpful options:

```bash
python -m book_writer generate books/my_new_book --chapters 1,2 --max-concurrent 3
python -m book_writer generate books/my_new_book --test-run
python -m book_writer generate books/my_new_book --phase1-model anthropic/claude-3.5-sonnet
```

4. **Resume, check status, and combine outputs**

```bash
python -m book_writer resume books/my_new_book
python -m book_writer status books/my_new_book
python -m book_writer combine books/my_new_book
```

The combined markdown is written to `output/book.md`. You can also convert to PDF/EPUB (requires Pandoc and the system
dependencies described above):

```bash
python -m book_writer convert books/my_new_book --format both
```

## LLM Provider Configuration

The system supports multiple LLM providers with direct SDK integration for better performance and cost efficiency:

- **OpenRouter**: API proxy for various models (default)
- **OpenAI**: Direct SDK integration using `openai` package
- **Gemini**: Direct SDK integration using `google-genai` package

### Environment Variables

Set the appropriate API keys for the providers you want to use:

```bash
# For OpenRouter models
export OPENROUTER_API_KEY="your_openrouter_key_here"

# For OpenAI models (direct SDK)
export OPENAI_API_KEY="your_openai_key_here"

# For Gemini models (direct SDK)
export GEMINI_API_KEY="your_gemini_key_here"
```

### Using Predefined Models

```python
from llm_core import GEMINI_FLASH, OPENAI_04_MINI, GROK_3_MINI, LlmModel

# Use Gemini directly via Google's SDK (faster, cheaper)
model = LlmModel.create(model=GEMINI_FLASH, temperature=0.2)

# Use OpenAI directly via OpenAI's SDK
model = LlmModel.create(model=OPENAI_04_MINI, temperature=0.2)

# Use OpenRouter (for models not available via direct SDKs)
model = LlmModel.create(model=GROK_3_MINI, temperature=0.2)
```

### Available Model Constants

**Gemini Models (Direct SDK)**
- `GEMINI_FLASH` → `gemini-2.5-flash`
- `GEMINI_PRO` → `gemini-2.5-pro`
- `GEMINI_FLASH_LITE` → `gemini-2.5-flash-lite`

**OpenAI Models (Direct SDK)**
- `OPENAI_04_MINI` → `o4-mini`

**OpenRouter Models**
- `GROK_3_MINI` → `x-ai/grok-3-mini`
- `DEEPSEEK` → `deepseek/deepseek-r1-0528`
- `CLAUDE_4_SONNET` → `anthropic/claude-sonnet-4`
- `KIMI_K2` → `moonshotai/kimi-k2:free`

### Custom Model Configurations

```python
from llm_core import LlmModel, ModelConfig, Provider

# Custom OpenAI model
custom_openai = ModelConfig(
    provider=Provider.OPENAI,
    model_id="gpt-4o",
    provider_model_name="gpt-4o"
)
model = LlmModel.create(model=custom_openai)

# Custom Gemini model
custom_gemini = ModelConfig(
    provider=Provider.GEMINI,
    model_id="gemini-pro",
    provider_model_name="gemini-pro"
)
model = LlmModel.create(model=custom_gemini)
```

### Configuration in Pipeline

In your pipeline configuration, use ModelConfig objects:

```python
from book_updater import PhaseConfig, PhaseType
from llm_core import GEMINI_FLASH

phase_config = PhaseConfig(
    phase_type=PhaseType.MODERNIZE,
    model=GEMINI_FLASH,  # Direct Gemini SDK
    temperature=0.2
)
```

Configuration models are Pydantic v2 models, so invalid inputs raise
`pydantic.ValidationError` during instantiation.

### Benefits of Direct SDK Integration

- **Performance**: Direct SDK calls are faster than OpenRouter proxy
- **Cost**: No OpenRouter markup fees for direct provider calls
- **Reliability**: Provider-specific retry logic and error handling
- **Features**: Access to provider-specific optimizations

## Cost Tracking

The system includes an optional cost tracking feature for LLM API usage across all providers. This allows you to monitor and log costs for each phase and the total run without modifying existing code.

### Quick Start with Cost Tracking

1. Set your API key for the provider you're using:
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"    # For OpenRouter
   export OPENAI_API_KEY="your-api-key-here"        # For OpenAI
   export GEMINI_API_KEY="your-api-key-here"        # For Gemini
   ```

2. Use the CostTrackingWrapper class in your code:
   ```python
   from llm_core.cost import CostTrackingWrapper

   # Initialize the wrapper with your API key
   cost_wrapper = CostTrackingWrapper(api_key="your-openrouter-api-key")
   # Or let it auto-detect from environment variables
   cost_wrapper = CostTrackingWrapper()

   # After each API call, add the generation ID
   processed_body, generation_id = self.model.chat_completion(...)
   cost_wrapper.add_generation_id(
       phase_name="phase_name",
       generation_id=generation_id,
       model="gpt-4o-mini",  # Optional for better accuracy
       prompt_tokens=100,    # Optional for cost estimation
       completion_tokens=50  # Optional for cost estimation
   )

   # At the end of your pipeline, calculate and log costs
   phase_names = ["modernize", "edit", "final"]
   run_costs = cost_wrapper.calculate_and_log_costs(phase_names)
   ```

3. Optional: Create your own cost tracking example by adding a file like `examples/cost_tracking_example.py` with the above code patterns.

## Command Line Interface

The project provides a unified command-line interface for building books and running pipeline processing.

### Available Commands

- **build**: Build books from markdown sources to EPUB/PDF formats
- **run**: Run pipeline processing for books from markdown sources

### Building Books

Use the build command to convert processed markdown files to EPUB and PDF formats:

```bash
python -m cli build <book_name> <version>
```

**Examples:**
```bash
python -m cli build the_federalist_papers v1.0.0
python -m cli build on_liberty v0.1-alpha
```

### Running Pipeline Processing

Use the run command to process books through the pipeline:

```bash
python -m cli run <book_name>
```

**Example:**
```bash
python -m cli run on_liberty
```

### Getting Help

To see all available commands and options:

```bash
python -m cli --help
```

To see help for a specific command:

```bash
python -m cli build --help
python -m cli run --help
```

### Legacy Build Commands

The old build commands still work for backward compatibility:
```bash
python -m build <book_name> <version>
python -m books.the_federalist_papers.build build v0.0-alpha
```

### Book Writer CLI

The book writer uses its own CLI module:

```bash
python -m book_writer --help
```

Common commands:

```bash
python -m book_writer init <book_dir> --title "Book Title"
python -m book_writer generate <book_dir>
python -m book_writer resume <book_dir>
python -m book_writer status <book_dir>
python -m book_writer combine <book_dir>
python -m book_writer convert <book_dir> --format both
```

## Examples

See the `examples/` directory for complete working examples:

-   `annotation_example.py`: Demonstrates all three phase types.
-   `run_pipeline_example.py`: Shows how to run the full pipeline.
-   `metadata_example.py`: Example of how metadata is saved.
-   `retry_example.py`: Example with retry logic.
-   `default_postprocessors_example.py`: Demonstrates default post-processor configurations.
