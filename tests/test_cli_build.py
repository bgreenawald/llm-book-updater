from click.testing import CliRunner

from cli.build import build_command


def test_build_command_keeps_existing_book_version_form(monkeypatch):
    calls = []

    monkeypatch.setattr("cli.build.get_books_with_build", lambda: ["on_liberty"])
    monkeypatch.setattr(
        "cli.build.build_book",
        lambda book_name, version, require_abridged=False: calls.append((book_name, version, require_abridged)),
    )

    result = CliRunner().invoke(build_command, ["on_liberty", "v1.0.0"])

    assert result.exit_code == 0
    assert calls == [("on_liberty", "v1.0.0", False)]
    assert "Building book 'on_liberty' version 'v1.0.0'..." in result.output


def test_build_abridged_command_uses_existing_build_flow(monkeypatch):
    calls = []

    monkeypatch.setattr("cli.build.get_books_with_build", lambda: ["on_liberty"])
    monkeypatch.setattr(
        "cli.build.build_book",
        lambda book_name, version, require_abridged=False: calls.append((book_name, version, require_abridged)),
    )

    result = CliRunner().invoke(build_command, ["abridged", "on_liberty", "v1.0.0"])

    assert result.exit_code == 0
    assert calls == [("on_liberty", "v1.0.0", True)]
    assert "Building abridged book 'on_liberty' version 'v1.0.0'..." in result.output


def test_build_legacy_form_help_does_not_treat_help_as_version(monkeypatch):
    calls = []

    monkeypatch.setattr("cli.build.build_book", lambda book_name, version: calls.append((book_name, version)))

    result = CliRunner().invoke(build_command, ["on_liberty", "--help"])

    assert result.exit_code == 0
    assert calls == []
    assert "Usage:" in result.output
    assert "abridged" in result.output
