import sys
from pathlib import Path
from types import SimpleNamespace

from books.base_builder import BaseBookBuilder, BookConfig


class AbridgedSourceBuilder(BaseBookBuilder):
    def get_source_files(self):
        return {}

    def get_original_file(self):
        return None


def test_get_abridged_file_uses_configured_final_output_not_stale_glob_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_dir = Path("books/test_book/output")
    output_dir.mkdir(parents=True)

    stale_output = output_dir / "08-input_transformed Abridge_write_1.md"
    final_output = output_dir / "03-01-input_transformed Modernize_1 Abridge_write_1.md"
    stale_output.write_text("stale", encoding="utf-8")
    final_output.write_text("final", encoding="utf-8")

    abridge_config = SimpleNamespace(
        input_file=output_dir / "01-input_transformed Modernize_1.md",
        output_dir=output_dir,
        phases=[
            SimpleNamespace(phase_type=SimpleNamespace(name="ABRIDGE_PLAN"), custom_output_path=None),
            SimpleNamespace(phase_type=SimpleNamespace(name="ABRIDGE_FLESH"), custom_output_path=None),
            SimpleNamespace(phase_type=SimpleNamespace(name="ABRIDGE_WRITE"), custom_output_path=None),
        ],
    )
    monkeypatch.setitem(sys.modules, "books.test_book.abridge", SimpleNamespace(config=abridge_config))

    config = BookConfig(name="test_book", version="v1", title="Test Book", author="Author")
    builder = AbridgedSourceBuilder(config)

    assert builder.get_abridged_file() == final_output


def test_get_abridged_file_requires_configured_final_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_dir = Path("books/test_book/output")
    output_dir.mkdir(parents=True)

    abridge_config = SimpleNamespace(
        input_file=output_dir / "01-input_transformed Modernize_1.md",
        output_dir=output_dir,
        phases=[
            SimpleNamespace(phase_type=SimpleNamespace(name="ABRIDGE_PLAN"), custom_output_path=None),
            SimpleNamespace(phase_type=SimpleNamespace(name="ABRIDGE_FLESH"), custom_output_path=None),
            SimpleNamespace(phase_type=SimpleNamespace(name="ABRIDGE_WRITE"), custom_output_path=None),
        ],
    )
    monkeypatch.setitem(sys.modules, "books.test_book.abridge", SimpleNamespace(config=abridge_config))
    monkeypatch.setenv("LLM_BOOK_BUILD_REQUIRE_ABRIDGED", "1")

    config = BookConfig(name="test_book", version="v1", title="Test Book", author="Author")
    builder = AbridgedSourceBuilder(config)

    expected_path = output_dir / "03-01-input_transformed Modernize_1 Abridge_write_1.md"
    try:
        builder.get_abridged_file()
    except FileNotFoundError as exc:
        assert str(expected_path) in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
