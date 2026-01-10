"""
Release command module.

This module handles creating GitHub releases for built books.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import click


def parse_build_folder(build_folder: str) -> tuple[str, str]:
    """
    Parse a build folder path to extract book name and version.

    Args:
        build_folder: Path to build folder (e.g., 'build/BOOK_NAME/VERSION')

    Returns:
        Tuple of (book_name, version)

    Raises:
        click.ClickException: If the path format is invalid
    """
    path = Path(build_folder)
    parts = path.parts

    # Handle both 'build/book_name/version' and 'book_name/version' formats
    if len(parts) >= 2:
        if parts[0] == "build" and len(parts) >= 3:
            book_name = parts[1]
            version = parts[2]
        elif len(parts) == 2:
            book_name = parts[0]
            version = parts[1]
        else:
            book_name = parts[-2]
            version = parts[-1]
        return book_name, version

    raise click.ClickException(
        f"Invalid build folder format: '{build_folder}'. "
        "Expected format: 'build/BOOK_NAME/VERSION' or 'BOOK_NAME/VERSION'"
    )


def get_existing_tags(book_name: str) -> List[str]:
    """
    Get all existing git tags that start with the book name.

    Args:
        book_name: Name of the book (with underscores)

    Returns:
        List of matching tag names
    """
    # Convert underscores to hyphens for tag matching
    tag_prefix = book_name.replace("_", "-")

    try:
        result = subprocess.run(
            ["git", "tag", "-l", f"{tag_prefix}--*"],
            capture_output=True,
            text=True,
            check=True,
        )
        tags = [tag.strip() for tag in result.stdout.strip().split("\n") if tag.strip()]
        return sorted(tags)
    except subprocess.CalledProcessError:
        return []


def get_build_assets(build_folder: str) -> List[Path]:
    """
    Get all files in the build folder that should be uploaded as release assets.

    Args:
        build_folder: Path to the build folder

    Returns:
        List of file paths
    """
    path = Path(build_folder)
    if not path.exists():
        raise click.ClickException(f"Build folder does not exist: '{build_folder}'")

    if not path.is_dir():
        raise click.ClickException(f"Build folder is not a directory: '{build_folder}'")

    # Get all files in the build folder (not recursive)
    return sorted([f for f in path.iterdir() if f.is_file()])


def create_release_tag(tag: str) -> bool:
    """
    Create a git tag.

    Args:
        tag: Tag name to create

    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(["git", "tag", tag], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        click.echo(f"Error creating tag: {e.stderr}")
        return False


def push_tag(tag: str) -> bool:
    """
    Push a git tag to origin.

    Args:
        tag: Tag name to push

    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(["git", "push", "origin", tag], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        click.echo(f"Error pushing tag: {e.stderr}")
        return False


def create_github_release(tag: str, assets: List[Path], release_notes: Optional[str] = None) -> bool:
    """
    Create a GitHub release using the gh CLI.

    Args:
        tag: Tag name for the release
        assets: List of file paths to upload as assets
        release_notes: Optional release notes

    Returns:
        True if successful, False otherwise
    """
    cmd = ["gh", "release", "create", tag]

    if release_notes:
        cmd.extend(["--notes", release_notes])
    else:
        cmd.append("--generate-notes")

    # Add all assets
    for asset in assets:
        cmd.append(str(asset))

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        click.echo(f"Error creating GitHub release: {e}")
        return False


@click.command("release")
@click.argument("build_folder", required=True)
@click.option(
    "--notes",
    "-n",
    type=str,
    default=None,
    help="Release notes for the GitHub release",
)
@click.help_option("--help", "-h")
def release_command(build_folder: str, notes: Optional[str]) -> None:
    """
    Create a GitHub release for a built book.

    BUILD_FOLDER: Path to the build folder (e.g., 'build/the_federalist_papers/v1.0')

    This command will:
    1. Parse the build folder to extract book name and version
    2. Create a git tag of the form BOOK_NAME--VERSION
    3. Create a GitHub release with all files in the build folder as assets

    Examples:
      python -m cli release build/the_federalist_papers/v1.0
      python -m cli release build/on_liberty/v2.0 --notes "Bug fixes and improvements"
    """
    # Parse the build folder
    book_name, version = parse_build_folder(build_folder)

    # Create tag name (convert underscores to hyphens)
    tag = f"{book_name.replace('_', '-')}--{version}"

    # Get existing tags for this book
    existing_tags = get_existing_tags(book_name)

    # Get assets to upload
    try:
        assets = get_build_assets(build_folder)
    except click.ClickException:
        raise

    if not assets:
        raise click.ClickException(f"No files found in build folder: '{build_folder}'")

    # Display confirmation information
    click.echo("\n" + "=" * 60)
    click.echo("RELEASE CONFIRMATION")
    click.echo("=" * 60)

    click.echo(f"\nTag to be created: {tag}")

    click.echo(f"\nExisting tags for '{book_name.replace('_', '-')}':")
    if existing_tags:
        for existing_tag in existing_tags:
            click.echo(f"  - {existing_tag}")
    else:
        click.echo("  (none)")

    click.echo(f"\nAssets to be uploaded ({len(assets)} files):")
    for asset in assets:
        click.echo(f"  - {asset.name}")

    if notes:
        click.echo(f"\nRelease notes: {notes}")
    else:
        click.echo("\nRelease notes: (auto-generated)")

    click.echo("\n" + "=" * 60)

    # Ask for confirmation
    if not click.confirm("\nDo you want to proceed with this release?"):
        click.echo("Release cancelled.")
        sys.exit(0)

    # Create the tag
    click.echo(f"\nCreating tag '{tag}'...")
    if not create_release_tag(tag):
        sys.exit(1)

    # Push the tag
    click.echo(f"Pushing tag '{tag}' to origin...")
    if not push_tag(tag):
        sys.exit(1)

    # Create the GitHub release
    click.echo("Creating GitHub release...")
    if not create_github_release(tag, assets, notes):
        sys.exit(1)

    click.echo(f"\nRelease '{tag}' created successfully!")
