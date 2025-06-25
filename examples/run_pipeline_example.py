"""
Example script demonstrating how to use the new pipeline system.
"""

from src.run_pipeline import create_default_run, run_pipeline


def main():
    # Create a default run configuration
    config = create_default_run(
        book_name="On Liberty",
        author_name="John Stuart Mill",
        input_file=r"books\On Liberty\markdown\Mill, On Liberty\Mill, On Liberty Clean.md",
        output_dir=r"books\On Liberty\markdown\Mill, On Liberty",
        custom_phases={
            # Example: Disable the edit phase
            # PhaseType.EDIT: {'enabled': False},
            # Example: Use a different model for annotation
            # PhaseType.ANNOTATE: {'model_type': ModelType.GEMINI_PRO},
            # Example: Set a custom temperature for a specific phase
            # PhaseType.MODERNIZE: {'temperature': 0.3},
            # Example: Set a custom output path for a phase
            # PhaseType.ANNOTATE: {
            #     'custom_output_path': Path("path/to/custom_annotated_output.md")
            # },
            # Example: Use a different model for formatting check
            # PhaseType.FORMATTING: {'model_type': ModelType.GEMINI_PRO},
        },
    )

    # Run the pipeline
    run_pipeline(config)


if __name__ == "__main__":
    main()
