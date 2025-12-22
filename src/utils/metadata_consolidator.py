"""
Metadata consolidation utility.

This module provides functionality to consolidate multiple pipeline metadata files
into a single comprehensive metadata file that shows all completed phases.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging_config import setup_logging

logger = setup_logging("metadata_consolidator")


class MetadataConsolidator:
    """Consolidates multiple pipeline metadata files into a single file."""

    def __init__(self, output_dir: Path):
        """
        Initialize the metadata consolidator.

        Args:
            output_dir: Directory containing metadata files to consolidate
        """
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            raise ValueError(f"Output directory does not exist: {output_dir}")

    def find_metadata_files(self) -> List[Path]:
        """
        Find all metadata JSON files in the output directory.

        Returns:
            List of paths to metadata files, sorted by modification time (oldest first)
        """
        metadata_files = list(self.output_dir.glob("pipeline_metadata_*.json"))
        # Sort by modification time (oldest first) to preserve chronological order
        metadata_files.sort(key=lambda p: p.stat().st_mtime)
        return metadata_files

    def load_metadata_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load a metadata JSON file.

        Args:
            file_path: Path to the metadata file

        Returns:
            Metadata dictionary, or None if file cannot be loaded
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load metadata file {file_path}: {e}")
            return None

    def consolidate_metadata(self, metadata_files: List[Path]) -> Dict[str, Any]:
        """
        Consolidate multiple metadata files into a single metadata structure.

        The consolidation strategy:
        1. Use the most recent metadata file as the base
        2. For each phase index, use the completed phase data if available
        3. Prefer completed phases over skipped/disabled phases
        4. Preserve the most recent run's metadata version and book info

        Args:
            metadata_files: List of metadata file paths to consolidate

        Returns:
            Consolidated metadata dictionary
        """
        if not metadata_files:
            raise ValueError("No metadata files provided for consolidation")

        # Load all metadata files
        all_metadata = []
        for file_path in metadata_files:
            metadata = self.load_metadata_file(file_path)
            if metadata:
                all_metadata.append(metadata)

        if not all_metadata:
            raise ValueError("No valid metadata files could be loaded")

        # Use the most recent metadata as the base
        base_metadata = all_metadata[-1].copy()

        # Get the maximum number of phases across all runs
        max_phases = max(len(m.get("phases", [])) for m in all_metadata)

        # Initialize consolidated phases list
        consolidated_phases: List[Optional[Dict[str, Any]]] = [None] * max_phases

        # Consolidate phase data
        for metadata in all_metadata:
            phases = metadata.get("phases", [])
            for phase in phases:
                phase_index = phase.get("phase_index")
                if phase_index is None:
                    continue

                # If this phase is completed, use it
                if phase.get("completed", False):
                    consolidated_phases[phase_index] = phase
                # If we don't have data for this phase yet, use whatever we have
                elif consolidated_phases[phase_index] is None:
                    consolidated_phases[phase_index] = phase

        # Filter out None values and create final phases list
        final_phases = [p for p in consolidated_phases if p is not None]

        # Update the base metadata with consolidated phases
        base_metadata["phases"] = final_phases

        # Add consolidation metadata
        base_metadata["consolidation_info"] = {
            "consolidated_at": datetime.now().isoformat(),
            "source_files": [f.name for f in metadata_files],
            "num_source_files": len(metadata_files),
        }

        # Update run timestamp to reflect consolidation
        base_metadata["consolidated_run_timestamp"] = base_metadata.get("run_timestamp")
        base_metadata["run_timestamp"] = datetime.now().isoformat()

        return base_metadata

    def save_consolidated_metadata(
        self, consolidated_metadata: Dict[str, Any], output_filename: str = "pipeline_metadata_consolidated.json"
    ) -> Path:
        """
        Save consolidated metadata to a file.

        Args:
            consolidated_metadata: The consolidated metadata dictionary
            output_filename: Name of the output file

        Returns:
            Path to the saved file
        """
        output_path = self.output_dir / output_filename

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(consolidated_metadata, f, indent=2, ensure_ascii=False)

            logger.success(f"Consolidated metadata saved to: {output_path}")
            return output_path

        except OSError as e:
            logger.error(f"Failed to save consolidated metadata: {e}")
            raise

    def run(self, output_filename: str = "pipeline_metadata_consolidated.json") -> Path:
        """
        Find, consolidate, and save metadata files.

        Args:
            output_filename: Name of the output file

        Returns:
            Path to the consolidated metadata file
        """
        # Find metadata files
        metadata_files = self.find_metadata_files()

        if not metadata_files:
            raise ValueError(f"No metadata files found in {self.output_dir}")

        logger.info(f"Found {len(metadata_files)} metadata files to consolidate")
        for file in metadata_files:
            logger.info(f"  - {file.name}")

        # Consolidate metadata
        consolidated_metadata = self.consolidate_metadata(metadata_files)

        # Count completed phases
        completed_count = sum(1 for phase in consolidated_metadata["phases"] if phase.get("completed", False))
        total_count = len(consolidated_metadata["phases"])

        logger.info(f"Consolidated {completed_count}/{total_count} phases")

        # Save consolidated metadata
        output_path = self.save_consolidated_metadata(consolidated_metadata, output_filename)

        return output_path


def consolidate_metadata(output_dir: Path, output_filename: str = "pipeline_metadata_consolidated.json") -> Path:
    """
    Convenience function to consolidate metadata files in a directory.

    Args:
        output_dir: Directory containing metadata files
        output_filename: Name of the output file

    Returns:
        Path to the consolidated metadata file
    """
    consolidator = MetadataConsolidator(output_dir)
    return consolidator.run(output_filename)
