# Book Converter

## Convert PDF to MD

```powershell
uv run marker_single /path/to/file.pdf --output_format markdown --output_dir PATH --use_llm --gemini_api_key GEMINI_API_KEY
```

## Pipeline Metadata

The pipeline automatically saves metadata about each run in the output directory. This includes:

- **Run timestamp**: When the pipeline was executed
- **Book information**: Book name and author
- **File paths**: Input, output, and original file locations
- **Phase details**: Configuration and completion status for each phase
- **Model settings**: Temperature, model type, and other parameters used

### Metadata File Format

Metadata is saved as JSON files with the naming pattern: `run_metadata_YYYYMMDD_HHMMSS.json`

Example metadata structure:
```json
{
  "run_timestamp": "2024-01-15T14:30:25.123456",
  "book_name": "Example Book",
  "author_name": "Example Author",
  "input_file": "/path/to/input.md",
  "original_file": "/path/to/original.pdf",
  "output_directory": "/path/to/output",
  "phases": {
    "MODERNIZE": {
      "enabled": true,
      "model_type": "gemini-flash",
      "temperature": 0.2,
      "input_file": "/path/to/input.md",
      "output_file": "/path/to/output/input Modernize.md",
      "system_prompt": "/path/to/prompts/modernize.md",
      "user_prompt": null,
      "max_workers": null,
      "completed": true,
      "output_exists": true
    },
    "EDIT": {
      "enabled": false,
      "model_type": "gemini-flash",
      "temperature": 0.2,
      "max_workers": null,
      "completed": false,
      "reason": "disabled"
    }
  }
}
```

### Usage

Metadata is automatically saved when:
- Running the complete pipeline with `pipeline.run()` - **saves metadata once for the entire run**
- Running individual phases with `pipeline.run_phase(phase_type, save_metadata=True)` - **optional metadata saving for single phases**

The metadata files are saved in the same output directory as the processed files, making it easy to track the history and configuration of each pipeline run.

**Note**: By default, metadata is only saved once per complete pipeline run to avoid creating too many files. If you need metadata for individual phase runs, explicitly set `save_metadata=True` when calling `run_phase()`.