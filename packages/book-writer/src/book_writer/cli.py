"""CLI interface for the book writer application."""

import asyncio
from pathlib import Path
from typing import Optional

import click
from llm_core import AsyncOpenRouterClient
from rich.console import Console
from rich.table import Table

from .config import (
    ensure_output_directory,
    get_api_key,
    get_generation_config,
    load_book_config,
    save_book_config,
    validate_book_directory,
)
from .generator import BookGenerator, combine_chapters
from .models import BookConfig, ChapterStatus
from .parser import compute_rubric_hash, parse_rubric
from .state import StateManager

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Business Book Writer - Generate book drafts from outlines using LLMs."""
    pass


@cli.command()
@click.argument("book_dir", type=click.Path(exists=False), required=True)
@click.option("--title", "-t", required=True, help="Book title")
@click.option("--model", "-m", default="anthropic/claude-sonnet-4", help="LLM model to use")
def init(book_dir: str, title: str, model: str):
    """Initialize a new book project."""
    book_path = Path(book_dir)

    if book_path.exists():
        console.print(f"[red]Directory already exists: {book_path}[/red]")
        return

    # Create directory structure
    book_path.mkdir(parents=True)
    (book_path / "output" / "chapters").mkdir(parents=True)

    # Create config.yaml
    config = BookConfig(title=title, model=model)
    save_book_config(book_path, config)

    # Create empty rubric.md with template
    rubric_path = book_path / "rubric.md"
    rubric_template = f"""# {title}

## Chapter Goals
- Define your chapter goals here

## 1.1 First Section

### Subsection guidance
- Add your outline content here

## 1.2 Second Section

### Subsection guidance
- Continue adding sections...
"""
    rubric_path.write_text(rubric_template, encoding="utf-8")

    console.print(f"[green]Created book project at: {book_path}[/green]")
    console.print("  - config.yaml: Book settings")
    console.print("  - rubric.md: Edit this file with your book outline")
    console.print("  - output/: Generated content will go here")


@cli.command()
@click.argument("book_dir", type=click.Path(exists=True), required=True)
@click.option("--chapters", "-c", help="Comma-separated chapter numbers to generate")
@click.option("--model", "-m", help="Override model from config")
@click.option("--max-concurrent", type=int, help="Max concurrent chapters")
def generate(
    book_dir: str,
    chapters: Optional[str],
    model: Optional[str],
    max_concurrent: Optional[int],
):
    """Generate book content from the rubric outline."""
    book_path = Path(book_dir)

    try:
        validate_book_directory(book_path)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return

    # Parse chapters option
    chapter_list = None
    if chapters:
        chapter_list = [c.strip() for c in chapters.split(",")]

    # Load configuration
    try:
        api_key = get_api_key()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return

    gen_config = get_generation_config(
        book_path,
        model_override=model,
        max_concurrent_override=max_concurrent,
    )

    # Parse rubric
    rubric_path = book_path / "rubric.md"
    outline = parse_rubric(rubric_path)
    rubric_hash = compute_rubric_hash(rubric_path)

    console.print(f"[blue]Book: {outline.title}[/blue]")
    console.print(f"[blue]Model: {gen_config.model}[/blue]")
    console.print(f"[blue]Chapters: {len(outline.chapters)}[/blue]")

    # Setup state
    output_dir = ensure_output_directory(book_path)
    state_manager = StateManager(output_dir)

    state = state_manager.load_state()

    if state is None or state_manager.should_reinitialize(state, rubric_hash):
        if state is not None:
            console.print("[yellow]Rubric changed, reinitializing state...[/yellow]")
        state = state_manager.initialize_state(outline, gen_config.model, rubric_hash)
        console.print("[green]Initialized fresh state[/green]")
    else:
        console.print("[green]Resuming from existing state[/green]")

    # Progress tracking
    def progress_callback(ch_id, sec_id, status, message=None):
        if sec_id:
            if status == "generating":
                console.print(f"  [cyan]Generating {ch_id}.{sec_id}...[/cyan]")
            elif status == "completed":
                console.print(f"  [green]Completed {ch_id}.{sec_id}[/green]")
            elif status == "failed":
                console.print(f"  [red]Failed {ch_id}.{sec_id}: {message}[/red]")
        else:
            if status == "started":
                console.print(f"[blue]Starting chapter {ch_id}[/blue]")
            elif status == "chapter_completed":
                console.print(f"[green]Completed chapter {ch_id}[/green]")
            elif status == "chapter_stopped":
                console.print(f"[yellow]Stopped chapter {ch_id}: {message}[/yellow]")

    # Run generation
    async def run():
        async with AsyncOpenRouterClient(
            api_key=api_key,
            default_model=gen_config.model,
            max_retries=gen_config.max_retries,
        ) as client:
            generator = BookGenerator(
                outline=outline,
                client=client,
                state_manager=state_manager,
                config=gen_config,
                output_dir=output_dir,
                progress_callback=progress_callback,
            )
            return await generator.generate_book(state, chapter_list)

    console.print("\n[bold]Starting generation...[/bold]\n")
    final_state = asyncio.run(run())

    # Show summary
    progress = state_manager.get_overall_progress(final_state)
    console.print("\n[bold]Generation complete![/bold]")
    console.print(f"  Sections completed: {progress['completed']}/{progress['total_sections']}")
    if progress["failed"] > 0:
        console.print(f"  [red]Sections failed: {progress['failed']}[/red]")
        console.print("  Run 'bookwriter resume' to retry failed sections")


@cli.command()
@click.argument("book_dir", type=click.Path(exists=True), required=True)
@click.option("--chapters", "-c", help="Comma-separated chapter numbers to retry")
def resume(book_dir: str, chapters: Optional[str]):
    """Resume generation of failed/incomplete sections."""
    book_path = Path(book_dir)

    try:
        validate_book_directory(book_path)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return

    output_dir = book_path / "output"
    state_manager = StateManager(output_dir)
    state = state_manager.load_state()

    if state is None:
        console.print("[red]No existing state found. Run 'generate' first.[/red]")
        return

    # Find sections needing work
    pending = state.get_pending_sections()

    if not pending:
        console.print("[green]All sections completed![/green]")
        return

    # Filter to specific chapters if requested
    if chapters:
        chapter_list = [c.strip() for c in chapters.split(",")]
        pending = [(ch, sec) for ch, sec in pending if ch in chapter_list]

    if not pending:
        console.print("[green]No pending sections in specified chapters.[/green]")
        return

    console.print(f"Found {len(pending)} sections to generate/retry")

    # Reset failed sections to pending
    state = state_manager.reset_failed_sections(state)

    # Get affected chapters
    affected_chapters = list(set(ch for ch, _ in pending))

    # Load configuration
    try:
        api_key = get_api_key()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return

    gen_config = get_generation_config(book_path)

    # Parse rubric
    rubric_path = book_path / "rubric.md"
    outline = parse_rubric(rubric_path)

    # Progress tracking
    def progress_callback(ch_id, sec_id, status, message=None):
        if sec_id:
            if status == "generating":
                console.print(f"  [cyan]Generating {ch_id}.{sec_id}...[/cyan]")
            elif status == "completed":
                console.print(f"  [green]Completed {ch_id}.{sec_id}[/green]")
            elif status == "failed":
                console.print(f"  [red]Failed {ch_id}.{sec_id}: {message}[/red]")

    # Run generation
    async def run():
        async with AsyncOpenRouterClient(
            api_key=api_key,
            default_model=gen_config.model,
            max_retries=gen_config.max_retries,
        ) as client:
            generator = BookGenerator(
                outline=outline,
                client=client,
                state_manager=state_manager,
                config=gen_config,
                output_dir=output_dir,
                progress_callback=progress_callback,
            )
            return await generator.generate_book(state, affected_chapters)

    console.print("\n[bold]Resuming generation...[/bold]\n")
    final_state = asyncio.run(run())

    # Show summary
    progress = state_manager.get_overall_progress(final_state)
    console.print("\n[bold]Resume complete![/bold]")
    console.print(f"  Sections completed: {progress['completed']}/{progress['total_sections']}")
    if progress["failed"] > 0:
        console.print(f"  [red]Sections still failed: {progress['failed']}[/red]")


@cli.command()
@click.argument("book_dir", type=click.Path(exists=True), required=True)
def status(book_dir: str):
    """Show current generation status."""
    book_path = Path(book_dir)
    output_dir = book_path / "output"
    state_manager = StateManager(output_dir)
    state = state_manager.load_state()

    if state is None:
        console.print("[yellow]No generation state found.[/yellow]")
        return

    # Load book config for title
    book_config = load_book_config(book_path)
    console.print(f"\n[bold]{book_config.title}[/bold]")
    console.print(f"Model: {state.model}")
    console.print(f"Created: {state.created_at}")
    console.print(f"Updated: {state.updated_at}\n")

    table = Table(title="Chapter Status")
    table.add_column("Chapter", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Sections", justify="right")
    table.add_column("Completed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Pending", justify="right", style="yellow")

    # Sort chapters: preface first, then numbered, then appendices
    def sort_key(ch_id):
        if ch_id == "preface":
            return (0, 0)
        elif ch_id.startswith("appendix_"):
            return (2, ord(ch_id[-1]))
        else:
            try:
                return (1, int(ch_id))
            except ValueError:
                return (1, 999)

    for ch_id in sorted(state.chapters.keys(), key=sort_key):
        ch_state = state.chapters[ch_id]
        progress = state_manager.get_chapter_progress(state, ch_id)

        # Format chapter name
        if ch_id == "preface":
            ch_name = "Preface"
        elif ch_id.startswith("appendix_"):
            ch_name = f"Appendix {ch_id[-1].upper()}"
        else:
            ch_name = f"Chapter {ch_id}"

        # Format status with color
        status_str = ch_state.status.value
        if ch_state.status == ChapterStatus.COMPLETED:
            status_str = f"[green]{status_str}[/green]"
        elif ch_state.status == ChapterStatus.FAILED:
            status_str = f"[red]{status_str}[/red]"
        elif ch_state.status == ChapterStatus.PARTIAL:
            status_str = f"[yellow]{status_str}[/yellow]"
        elif ch_state.status == ChapterStatus.IN_PROGRESS:
            status_str = f"[blue]{status_str}[/blue]"

        table.add_row(
            ch_name,
            status_str,
            str(progress["total"]),
            str(progress["completed"]),
            str(progress["failed"]),
            str(progress["pending"]),
        )

    console.print(table)

    # Overall summary
    overall = state_manager.get_overall_progress(state)
    console.print("\n[bold]Overall Progress:[/bold]")
    console.print(
        f"  {overall['completed']}/{overall['total_sections']} sections completed "
        f"({100 * overall['completed'] // max(1, overall['total_sections'])}%)"
    )
    if overall["failed"] > 0:
        console.print(f"  [red]{overall['failed']} sections failed[/red]")


@cli.command()
@click.argument("book_dir", type=click.Path(exists=True), required=True)
def combine(book_dir: str):
    """Combine all chapter files into a single book.md."""
    book_path = Path(book_dir)
    output_dir = book_path / "output"

    # Parse rubric for title
    rubric_path = book_path / "rubric.md"
    if not rubric_path.exists():
        console.print(f"[red]Rubric not found: {rubric_path}[/red]")
        return

    outline = parse_rubric(rubric_path)

    book_md = combine_chapters(output_dir, outline)
    console.print(f"[green]Created: {book_md}[/green]")


@cli.command()
@click.argument("book_dir", type=click.Path(exists=True), required=True)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["pdf", "epub", "both"]),
    default="both",
    help="Output format",
)
def convert(book_dir: str, format: str):
    """Convert generated markdown to PDF/EPUB using Pandoc."""
    from .converter import ConversionError, convert_to_epub, convert_to_pdf

    book_path = Path(book_dir)
    output_dir = book_path / "output"
    book_md = output_dir / "book.md"

    if not book_md.exists():
        # Try to combine first
        rubric_path = book_path / "rubric.md"
        if not rubric_path.exists():
            console.print("[red]No book.md or rubric.md found[/red]")
            return

        outline = parse_rubric(rubric_path)
        book_md = combine_chapters(output_dir, outline)
        console.print(f"[green]Combined chapters into: {book_md}[/green]")

    if format in ("pdf", "both"):
        try:
            pdf_path = convert_to_pdf(book_md, output_dir / "book.pdf")
            console.print(f"[green]Created: {pdf_path}[/green]")
        except ConversionError as e:
            console.print(f"[red]PDF conversion failed: {e}[/red]")

    if format in ("epub", "both"):
        try:
            epub_path = convert_to_epub(book_md, output_dir / "book.epub")
            console.print(f"[green]Created: {epub_path}[/green]")
        except ConversionError as e:
            console.print(f"[red]EPUB conversion failed: {e}[/red]")


@cli.command("list")
@click.argument("books_dir", type=click.Path(exists=True), required=True)
def list_books(books_dir: str):
    """List all book projects in a directory."""
    books_path = Path(books_dir)

    table = Table(title="Book Projects")
    table.add_column("Directory", style="cyan")
    table.add_column("Title", style="magenta")
    table.add_column("Status", justify="right")
    table.add_column("Progress", justify="right")

    for book_dir in sorted(books_path.iterdir()):
        if not book_dir.is_dir():
            continue

        rubric_path = book_dir / "rubric.md"
        if not rubric_path.exists():
            continue

        # Load config
        config = load_book_config(book_dir)

        # Check state
        output_dir = book_dir / "output"
        state_manager = StateManager(output_dir)
        state = state_manager.load_state()

        if state:
            progress = state_manager.get_overall_progress(state)
            total = progress["total_sections"]
            completed = progress["completed"]
            pct = 100 * completed // max(1, total)
            progress_str = f"{completed}/{total} ({pct}%)"

            # Determine status
            if completed == total and total > 0:
                status_str = "[green]Complete[/green]"
            elif progress["failed"] > 0:
                status_str = "[red]Has failures[/red]"
            elif completed > 0:
                status_str = "[yellow]In progress[/yellow]"
            else:
                status_str = "[blue]Not started[/blue]"
        else:
            status_str = "[dim]Not started[/dim]"
            progress_str = "-"

        table.add_row(book_dir.name, config.title, status_str, progress_str)

    console.print(table)


if __name__ == "__main__":
    cli()
