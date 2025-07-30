"""
Entry point for running the CLI as a module.

This allows the CLI to be executed with:
    python -m cli
"""

import os
import sys

# Add the parent directory to the path so we can import the main cli module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    import cli as main_cli

    main_cli.main()
