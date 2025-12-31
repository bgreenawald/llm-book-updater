"""
Consolidate metadata command module.

This module handles consolidating multiple pipeline metadata files into a single file.
"""

import sys
from pathlib import Path

import click
from book_updater.metadata_consolidator import consolidate_metadata


@click.command("consolidate-metadata")
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option(
    "--output-filename",
    type=str,
    default="pipeline_metadata_consolidated.json",
    help="Name of the consolidated metadata file.",
)
@click.help_option("--help", "-h")
def consolidate_command(output_dir: Path, output_filename: str) -> None:
    """
    Consolidate multiple pipeline metadata files into a single file.

    This is useful when you've run the pipeline multiple times with different
    start_from_phase values and want to see the complete picture of all phases
    that were successfully completed across all runs.

    OUTPUT_DIR: Directory containing the metadata files to consolidate

    Examples:
      python -m cli consolidate-metadata books/on_liberty/output
      python -m cli consolidate-metadata books/on_liberty/output --output-filename final_metadata.json
    """
    try:
        click.echo(f"Consolidating metadata files in: {output_dir}")

        # Consolidate metadata
        output_path = consolidate_metadata(output_dir, output_filename)

        click.echo(f"✓ Consolidated metadata saved to: {output_path}")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
