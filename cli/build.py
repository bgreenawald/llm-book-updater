"""
Build command module.

This module handles building books from markdown sources to EPUB/PDF formats.
"""

import importlib
import os
import sys

import click

from .common import get_books_with_build, validate_book_exists


class LegacyBuildGroup(click.Group):
    """Allow `build BOOK VERSION` while also supporting nested build subcommands."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and any(arg in {"--help", "-h"} for arg in args):
            click.echo(ctx.get_help())
            ctx.exit()

        if args and args[0] not in self.commands and args[0] not in {"--help", "-h"}:
            ctx.args = list(args)
            return []

        return super().parse_args(ctx, args)


def build_book(book_name: str, version: str, require_abridged: bool = False) -> None:
    """
    Build a specific book by importing its build module and calling the build function.

    Args:
        book_name: Name of the book to build
        version: Version string for the build
        require_abridged: Whether to fail if the final abridged manuscript is missing

    Raises:
        ImportError: If the book's build module cannot be imported
        AttributeError: If the book's build module doesn't have a build function
    """
    try:
        previous_require_abridged = os.environ.get("LLM_BOOK_BUILD_REQUIRE_ABRIDGED")
        if require_abridged:
            os.environ["LLM_BOOK_BUILD_REQUIRE_ABRIDGED"] = "1"

        try:
            # Import the book's build module
            module_path = f"books.{book_name}.build"
            build_module = importlib.import_module(module_path)

            # Call the build function
            if hasattr(build_module, "build"):
                build_module.build(version, book_name)
            else:
                click.echo(f"Error: Book '{book_name}' does not have a build function in its build.py module")
                sys.exit(1)
        finally:
            if previous_require_abridged is None:
                os.environ.pop("LLM_BOOK_BUILD_REQUIRE_ABRIDGED", None)
            else:
                os.environ["LLM_BOOK_BUILD_REQUIRE_ABRIDGED"] = previous_require_abridged

    except ImportError as e:
        click.echo(f"Error: Could not import build module for book '{book_name}': {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error building book '{book_name}': {e}")
        sys.exit(1)


def _run_build_command(book_name: str, version: str, label: str = "book", require_abridged: bool = False) -> None:
    available_books = get_books_with_build()

    if not available_books:
        click.echo("No buildable books found")
        sys.exit(1)

    matched_book_name = validate_book_exists(book_name, available_books)

    click.echo(f"Building {label} '{matched_book_name}' version '{version}'...")
    build_book(matched_book_name, version, require_abridged=require_abridged)


@click.group(
    "build",
    cls=LegacyBuildGroup,
    invoke_without_command=True,
)
@click.pass_context
@click.help_option("--help", "-h")
def build_command(ctx: click.Context) -> None:
    """
    Build books from markdown sources to EPUB/PDF formats.

    BOOK_NAME: Name of the book to build (must match a directory in books/)
    VERSION: Version string for the build (e.g., 'v1.0.0', 'v0.1-alpha')

    Examples:
      python -m cli build the_federalist_papers v1.0.0
      python -m cli build abridged the_federalist_papers v1.0.0
      python -m cli build on_liberty v0.1-alpha
    """
    if ctx.invoked_subcommand is not None:
        return

    if len(ctx.args) != 2:
        raise click.UsageError("Expected BOOK_NAME and VERSION, or use: build abridged BOOK_NAME VERSION")

    book_name, version = ctx.args
    _run_build_command(book_name, version)


@build_command.command("abridged")
@click.argument("book_name", required=True)
@click.argument("version", required=True)
@click.help_option("--help", "-h")
def build_abridged_command(book_name: str, version: str) -> None:
    """
    Build a book release that includes the generated abridged edition.

    BOOK_NAME: Name of the book to build (must match a directory in books/)
    VERSION: Version string for the build (e.g., 'v1.0.0', 'v0.1-alpha')

    Examples:
      python -m cli build abridged on_liberty v1.0.0
      python -m cli build abridged a_theory_of_justice v0.1-alpha
    """
    _run_build_command(book_name, version, label="abridged book", require_abridged=True)
