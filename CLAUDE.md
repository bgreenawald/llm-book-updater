# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Book Updater is a Python tool for processing and modernizing book content using Large Language Models. The system processes markdown files through configurable "phases" that can modernize language, edit content, add annotations, and generate additional sections like introductions and summaries.

## Architecture

### Core Components

- **Pipeline System** (`src/pipeline.py`): Orchestrates sequential execution of LLM processing phases
- **Phase Framework** (`src/llm_phase.py`): Abstract base class for all processing phases
- **Configuration** (`src/config.py`): Defines `PhaseConfig`, `RunConfig`, and enums for phase types and post-processors
- **Phase Factory** (`src/phase_factory.py`): Factory for creating phase instances with proper configuration
- **LLM Models** (`src/llm_model.py`): Wrapper for various LLM providers (OpenAI, Gemini, etc.)
- **Post-Processing** (`src/post_processors.py`): Chain of content processors for formatting and cleanup

### Processing Flow

1. **Input**: Markdown files (typically converted from PDF using marker tool)
2. **Phase Execution**: Sequential processing through configured phases (modernize, edit, annotate, etc.)
3. **Post-Processing**: Content cleanup and formatting
4. **Output**: Processed markdown files in `output/` directory with timestamped metadata

### Book Structure

Each book is organized in `books/<book_name>/` with:
- `input_raw.md`: Original content
- `input_small.md`: Reduced content for processing
- `input_transformed.md`: Preprocessed content
- `run.py`: Book-specific pipeline runner
- `build.py`: Book building configuration
- `output/`: Generated files and metadata

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
# Run all tests (note: some tests may have import issues)
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_performance.py

# Run performance tests only
uv run pytest tests/test_performance.py --collect-only
```

### Linting and Formatting
```bash
# Run ruff linting
uv run ruff check

# Fix linting issues
uv run ruff check --fix

# Format code
uv run ruff format

# Type checking (if mypy available)
uv run mypy src/

# Security scanning
uv run bandit -r . -x tests/
```

### CLI Usage
```bash
# Build a book to EPUB/PDF
python -m cli build <book_name> <version>
python -m cli build the_federalist_papers v1.0.0

# Run pipeline processing
python -m cli run <book_name>
python -m cli run on_liberty

# Get help
python -m cli --help
python -m cli build --help
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

## Key Architectural Patterns

### Phase System
Phases inherit from `LlmPhase` and implement specific processing logic. Available phase types:
- `MODERNIZE`: Update archaic language
- `EDIT`: General content editing
- `ANNOTATE`: Add annotations without changing original text
- `FINAL`: Final processing and cleanup
- `INTRODUCTION`: Generate introductions
- `SUMMARY`: Generate summaries

### Configuration-Driven Design
The system uses dataclasses (`PhaseConfig`, `RunConfig`) for type-safe configuration. Phases are configured declaratively and instantiated through the `PhaseFactory`.

### Post-Processing Pipeline
Content goes through configurable post-processors for:
- Formatting cleanup
- XML tag removal
- Content preservation
- Quote ordering

### Cost Tracking
Optional OpenRouter API cost tracking with generation ID collection and detailed reporting.

### Metadata Management
Comprehensive metadata collection including:
- Pipeline execution details
- Phase configurations
- Cost analysis
- Processing timestamps

## Important Notes

- The project uses `uv` as the package manager
- Tests may have import path issues requiring PYTHONPATH adjustment
- Book processing can be expensive - use cost tracking for monitoring
- System prompts are in `prompts/` directory with naming convention `{phase}_system.md`
- All LLM interactions go through the `LlmModel` wrapper for consistency
