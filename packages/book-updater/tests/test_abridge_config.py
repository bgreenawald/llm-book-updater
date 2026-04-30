"""Tests for shared abridge pipeline configuration."""

from book_updater.abridge_config import (
    DEFAULT_ABRIDGE_MAX_WORKERS,
    create_abridge_run_config,
    default_abridge_phases,
)
from book_updater.config import PhaseType


def test_default_abridge_phases_use_shared_models() -> None:
    phases = default_abridge_phases()

    assert [phase.phase_type for phase in phases] == [
        PhaseType.ABRIDGE_PLAN,
        PhaseType.ABRIDGE_FLESH,
        PhaseType.ABRIDGE_WRITE,
    ]
    assert phases[0].model.model_id == "openai/gpt-5.5"
    assert phases[0].reasoning == {"effort": "medium"}
    assert phases[1].model.model_id == "moonshotai/kimi-k2.6"
    assert phases[2].abridge_write_config is not None
    assert phases[2].abridge_write_config.profile_model.model_id == "openai/gpt-5.5"
    assert phases[2].abridge_write_config.write_model.model_id == "moonshotai/kimi-k2-thinking"


def test_create_abridge_run_config_uses_default_paths_and_workers(tmp_path) -> None:
    output_dir = tmp_path / "output"

    config = create_abridge_run_config(
        book_id="test_book",
        book_name="Test Book",
        author_name="Test Author",
        output_dir=output_dir,
    )

    assert config.input_file == output_dir / "01-input_transformed Modernize_1.md"
    assert config.original_file == config.input_file
    assert config.max_workers == DEFAULT_ABRIDGE_MAX_WORKERS
    assert len(config.phases) == 3


def test_create_abridge_run_config_allows_phase_overrides(tmp_path) -> None:
    config = create_abridge_run_config(
        book_id="test_book",
        book_name="Test Book",
        author_name="Test Author",
        output_dir=tmp_path / "output",
        max_workers=2,
        plan_overrides={"reasoning": {"effort": "high"}},
    )

    assert config.max_workers == 2
    assert config.phases[0].reasoning == {"effort": "high"}


def test_create_abridge_run_config_preserves_explicit_empty_phases(tmp_path) -> None:
    config = create_abridge_run_config(
        book_id="test_book",
        book_name="Test Book",
        author_name="Test Author",
        output_dir=tmp_path / "output",
        phases=[],
    )

    assert config.phases == []
