from pathlib import Path

from PIL import Image

from books.base_builder import BaseBookBuilder, BookConfig


class AbridgedCoverBuilder(BaseBookBuilder):
    def get_source_files(self):
        return {}

    def get_original_file(self):
        return None


def write_test_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), (255, 255, 255)).save(path)


def test_abridged_epub_uses_abridged_cover(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    book_dir = Path("books/test_book")
    write_test_image(book_dir / "cover.png")
    write_test_image(book_dir / "abridge-cover.png")

    config = BookConfig(name="test_book", version="v1", title="Test Book", author="Author")
    config.staging_dir.mkdir(parents=True)
    config.build_dir.mkdir(parents=True)
    config.staged_modernized_md.write_text("# Modernized\n", encoding="utf-8")
    config.staged_annotated_md.write_text("# Annotated\n", encoding="utf-8")
    config.staged_abridged_md.write_text("# Abridged\n", encoding="utf-8")

    builder = AbridgedCoverBuilder(config)
    convert_calls = []

    def fake_convert_file(source, output_format, outputfile, extra_args):
        convert_calls.append(
            {
                "source": source,
                "output_format": output_format,
                "outputfile": outputfile,
                "extra_args": extra_args,
            }
        )

    monkeypatch.setattr("books.base_builder.pypandoc.convert_file", fake_convert_file)
    monkeypatch.setattr(AbridgedCoverBuilder, "_build_pdf_from_epub", lambda *args, **kwargs: True)

    builder.build_epub_and_pdf()

    abridged_call = next(call for call in convert_calls if call["source"].endswith("abridged.md"))
    cover_args = [arg for arg in abridged_call["extra_args"] if arg.startswith("--epub-cover-image=")]

    assert cover_args == ["--epub-cover-image=books/test_book/staging/abridge-cover.jpg"]


def test_abridged_build_requires_abridged_cover(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    book_dir = Path("books/test_book")
    write_test_image(book_dir / "cover.png")

    config = BookConfig(name="test_book", version="v1", title="Test Book", author="Author")
    config.staging_dir.mkdir(parents=True)
    config.build_dir.mkdir(parents=True)
    config.staged_modernized_md.write_text("# Modernized\n", encoding="utf-8")
    config.staged_annotated_md.write_text("# Annotated\n", encoding="utf-8")
    config.staged_abridged_md.write_text("# Abridged\n", encoding="utf-8")

    builder = AbridgedCoverBuilder(config)
    monkeypatch.setenv("LLM_BOOK_BUILD_REQUIRE_ABRIDGED", "1")
    monkeypatch.setattr("books.base_builder.pypandoc.convert_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(AbridgedCoverBuilder, "_build_pdf_from_epub", lambda *args, **kwargs: True)

    try:
        builder.build_epub_and_pdf()
    except FileNotFoundError as exc:
        assert "Abridged cover not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
