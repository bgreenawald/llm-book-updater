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

# Import and run the main CLI
if __name__ == "__main__":
    import importlib.util

    spec = importlib.util.spec_from_file_location("main_cli", os.path.join(parent_dir, "cli.py"))
    main_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_cli)
    main_cli.main()
