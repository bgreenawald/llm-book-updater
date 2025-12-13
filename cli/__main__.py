"""
Entry point for running the CLI as a module.

This allows the CLI to be executed with:
    python -m cli
"""

import sys


def _main() -> None:
    """Run the CLI."""
    # Import lazily to avoid importing click/subcommands at module import time.
    from . import main

    main()


if __name__ == "__main__":
    # This file is intended to be executed as a package via `python -m cli`.
    # If executed directly (`python cli/__main__.py`), relative imports won't work.
    if not __package__:
        print(
            "Error: run the CLI as `python -m cli` (or install the package and use an entry point).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    _main()
