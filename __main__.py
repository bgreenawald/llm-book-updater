"""
Entry point for running the CLI module directly.

This allows the CLI to be run as:
    python -m llm-book-updater <command> [args...]

For backwards compatibility, also supports:
    python -m build <book_name> <version>
    python -m run <book_name>
"""

import sys

# Check for backwards compatibility with old build/run commands
if len(sys.argv) >= 2:
    # If called as package, first arg would be the command
    # Handle legacy usage patterns
    if "build.py" in sys.argv[0] or (len(sys.argv) >= 3 and sys.argv[1] not in ["build", "run", "--help", "-h"]):
        # Legacy build command: python -m build <book> <version>

        from cli.build import handle_build_command

        class Args:
            def __init__(self, book_name, version):
                self.book_name = book_name
                self.version = version

        if len(sys.argv) >= 3:
            args = Args(sys.argv[1], sys.argv[2])
            handle_build_command(args)
            sys.exit()
    elif "run.py" in sys.argv[0] or (len(sys.argv) >= 2 and sys.argv[1] not in ["build", "run", "--help", "-h"]):
        # Legacy run command: python -m run <book>

        from cli.run import handle_run_command

        class Args:
            def __init__(self, book_name):
                self.book_name = book_name

        if len(sys.argv) >= 2:
            args = Args(sys.argv[1])
            handle_run_command(args)
            sys.exit()

# Default to new CLI interface
from cli import main

if __name__ == "__main__":
    main()
