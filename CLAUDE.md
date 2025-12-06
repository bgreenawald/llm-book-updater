# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Book Updater is a Python tool for processing and modernizing book content using Large Language Models. The system processes markdown files through configurable "phases" that can modernize language, edit content, add annotations, and generate additional sections like introductions and summaries.

## Architecture

### Core Components

- **Pipeline System** (`src/pipeline.py`): Orchestrates sequential execution of LLM processing phases with model instance caching for connection pooling
- **Phase Framework** (`src/llm_phase.py`): Abstract base class for all processing phases with parallel processing support via ThreadPoolExecutor
- **Configuration** (`src/config.py`): Defines `PhaseConfig`, `RunConfig`, and enums for `PhaseType` and `PostProcessorType`
- **Phase Factory** (`src/phase_factory.py`): Factory for creating phase instances with proper configuration and post-processor chains
- **LLM Models** (`src/llm_model.py`): Provider-agnostic wrapper supporting OpenAI, Gemini, and OpenRouter with direct SDK integration
- **Post-Processing** (`src/post_processors.py`): Configurable chain of content processors with metadata tracking
- **Cost Tracking** (`src/cost_tracking_wrapper.py`, `src/cost_tracker.py`): Optional cost tracking for LLM API usage

### Processing Flow

1. **Input**: Markdown files (typically converted from PDF using `marker` tool)
2. **Phase Execution**: Sequential processing through configured phases with parallel block processing within each phase
3. **Post-Processing**: Content cleanup via configurable post-processor chains
4. **Output**: Processed markdown files in `output/` directory with comprehensive JSON metadata

### Book Structure

Each book is organized in `books/<book_name>/`:
- `input_raw.md`: Original content
- `input_small.md`: Reduced content for testing
- `input_transformed.md`: Preprocessed content (optional)
- `run.py`: Book-specific pipeline runner
- `build.py`: Book building configuration with metadata
- `output/`: Generated files and timestamped metadata

## Development Commands

### Setup
```bash
# Install dependencies
uv pip install .

# Install development dependencies
uv pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_configuration.py

# Run with verbose output
uv run pytest tests/test_performance.py -v

# Run tests with coverage
uv run pytest --cov=src tests/
```

**Note**: Some tests may require `PYTHONPATH` adjustment. Use:
```bash
PYTHONPATH=/home/ben-greenawald/Documents/llm-book-updater uv run pytest tests/test_system_prompt_metadata.py -v
```

### Linting and Formatting
```bash
# Run ruff linting
uv run ruff check

# Fix linting issues automatically
uv run ruff check --fix

# Format code
uv run ruff format

# Type checking
uv run mypy src/

# Security scanning
uv run bandit -r . -x tests/
```

**Ruff Configuration**: Line length is 120 characters (see `pyproject.toml`).

### CLI Usage
```bash
# Build a book to EPUB/PDF
python -m cli build <book_name> <version>
python -m cli build the_federalist_papers v1.0.0

# Run pipeline processing
python -m cli run <book_name>
python -m cli run on_liberty

# Consolidate metadata
python -m cli consolidate <book_name>

# Get help
python -m cli --help
python -m cli build --help
```

### Pre-commit Hooks
The project uses pre-commit hooks for:
- trailing-whitespace, end-of-file-fixer
- check-yaml, check-toml, check-merge-conflict
- ruff (linting and formatting)
- mypy (type checking)
- bandit (security scanning)
- pytest (test execution)

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

## Key Architectural Patterns

### Phase System
Phases inherit from `LlmPhase` (abstract base class) and implement specific processing logic. Available phase types in `PhaseType` enum:
- `MODERNIZE`: Update archaic language
- `EDIT`: General content editing
- `ANNOTATE`: Add annotations without changing original text
- `FINAL`: Final processing and cleanup
- `INTRODUCTION`: Generate introductions
- `SUMMARY`: Generate summaries

**Key Phase Implementation Details**:
- Phases process content in blocks (separated by `## ` headers)
- Parallel processing within phases via `ThreadPoolExecutor` (configurable `max_workers`)
- Optional batch processing for supported models (`use_batch`, `batch_size`)
- Post-processor chains applied after LLM processing

### Configuration-Driven Design
The system uses dataclasses (`PhaseConfig`, `RunConfig`) for type-safe configuration:
- `PhaseConfig`: Per-phase settings including model, temperature, prompts, post-processors
- `RunConfig`: Pipeline-level settings including phases, length reduction, tags to preserve, `start_from_phase` for resuming

**Important**: Phases are configured declaratively and instantiated through `PhaseFactory`.

### Post-Processing Pipeline
Content goes through configurable post-processor chains defined in `PostProcessorType` enum:
- **Formatting**: `ENSURE_BLANK_LINE`, `REMOVE_TRAILING_WHITESPACE`, `REMOVE_XML_TAGS`, `REMOVE_BLANK_LINES_IN_LIST`
- **Content Preservation**: `NO_NEW_HEADERS`, `REVERT_REMOVED_BLOCK_LINES`, `PRESERVE_F_STRING_TAGS`
- **Specialized**: `ORDER_QUOTE_ANNOTATION`

Post-processors can be specified as strings, `PostProcessorType` enums, or custom `PostProcessor` instances.

### Model Instance Caching
The Pipeline implements model instance caching (`_get_or_create_model`) to reuse connection pools across phases. Models are cached by `provider:model_id:temperature`.

### LLM Provider Integration
The system supports multiple providers via direct SDK integration:
- **OpenAI**: Using `openai` package (set `OPENAI_API_KEY`)
- **Gemini**: Using `google-genai` package (set `GEMINI_API_KEY`)
- **OpenRouter**: API proxy for various models (set `OPENROUTER_API_KEY`)

Predefined model constants in `src/llm_model.py`: `GEMINI_FLASH`, `GEMINI_PRO`, `OPENAI_04_MINI`, `GROK_3_MINI`, etc.

### Cost Tracking
Optional cost tracking via `CostTrackingWrapper`:
- Collects generation IDs from LLM API calls
- Queries OpenRouter API for detailed cost information
- Logs per-phase and total run costs
- Stored in metadata JSON files

### Metadata Management
Comprehensive metadata collection in `_phase_metadata`:
- Pipeline execution details
- Phase configurations (model, temperature, prompts)
- System prompt content (stored in metadata)
- Cost analysis (when cost tracking enabled)
- Processing timestamps
- Metadata version: `"0.0"` (defined in `src/pipeline.py`)

## Important Notes

- **Package Manager**: Project uses `uv` for dependency management
- **Python Version**: Requires Python >=3.12
- **System Prompts**: Located in `prompts/` directory with naming convention `{phase_type}_system.md` and `{phase_type}_user.md`
- **Content Splitting**: Content is split by `## ` (markdown level-2 headers) into blocks for parallel processing
- **Tags to Preserve**: F-string style tags like `{preface}`, `{license}` are preserved during processing (configurable via `tags_to_preserve`)
- **Length Reduction**: Optional parameter to guide LLM on content reduction (int percentage or tuple of min/max bounds)
- **Examples Directory**: Contains working examples for common use cases (annotation, metadata, retry logic, etc.)
- **EPUB/PDF Generation**: Requires Calibre for best results (working table of contents, proper formatting)
