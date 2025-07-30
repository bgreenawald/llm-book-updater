"""
Main CLI entry point for LLM Book Updater.

This module provides a unified command-line interface with subcommands for
different operations like building and running books.

Usage:
    python -m cli <command> [args...]
    python -m cli --help

Available commands:
    build   Build books from markdown sources to EPUB/PDF formats
    run     Run pipeline processing for books from markdown sources

Examples:
    python -m cli build the_federalist_papers v1.0.0
    python -m cli run on_liberty
"""

import argparse
import sys

from cli.build import handle_build_command, setup_build_parser
from cli.run import handle_run_command, setup_run_parser


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser with subcommands.

    Returns:
        The configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="cli",
        description="LLM Book Updater - A tool for processing and building books from markdown sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="<command>",
    )

    # Set up individual command parsers
    build_parser = setup_build_parser(subparsers)
    build_parser.set_defaults(func=handle_build_command)

    run_parser = setup_run_parser(subparsers)
    run_parser.set_defaults(func=handle_run_command)

    return parser


def main() -> None:
    """
    Main function to parse arguments and dispatch to appropriate command handler.
    """
    parser = create_parser()
    args = parser.parse_args()

    # If no command is provided, show help
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    # Call the appropriate command handler
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()