from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from llm_core import ModelConfig, Provider

from book_updater.config import AbridgeWriteModelConfig, PhaseConfig, PhaseType, RunConfig

GPT_55 = ModelConfig(provider=Provider.OPENROUTER, model_id="openai/gpt-5.5")
KIMI_K2 = ModelConfig(provider=Provider.OPENROUTER, model_id="moonshotai/kimi-k2-thinking")
KIMI_K2_6 = ModelConfig(provider=Provider.OPENROUTER, model_id="moonshotai/kimi-k2.6")

DEFAULT_ABRIDGE_MAX_WORKERS = 10
DEFAULT_ABRIDGE_MODERNIZED_FILENAME = "01-input_transformed Modernize_1.md"


def default_abridge_phases(
    *,
    plan_overrides: Mapping[str, Any] | None = None,
    flesh_overrides: Mapping[str, Any] | None = None,
    write_overrides: Mapping[str, Any] | None = None,
    write_config_overrides: Mapping[str, Any] | None = None,
) -> list[PhaseConfig]:
    """Return the default abridge phase pipeline with optional per-phase overrides."""
    write_config_kwargs: dict[str, Any] = {
        "profile_model": GPT_55,
        "write_model": KIMI_K2,
        "profile_reasoning": {"effort": "low"},
        "write_reasoning": {"effort": "high"},
    }
    if write_config_overrides:
        write_config_kwargs.update(write_config_overrides)

    plan_kwargs: dict[str, Any] = {
        "phase_type": PhaseType.ABRIDGE_PLAN,
        "model": GPT_55,
        "reasoning": {"effort": "medium"},
        "enable_retry": True,
    }
    flesh_kwargs: dict[str, Any] = {
        "phase_type": PhaseType.ABRIDGE_FLESH,
        "model": KIMI_K2_6,
        "reasoning": {"effort": "high"},
        "enable_retry": True,
    }
    write_kwargs: dict[str, Any] = {
        "phase_type": PhaseType.ABRIDGE_WRITE,
        "abridge_write_config": AbridgeWriteModelConfig(**write_config_kwargs),
        "enable_retry": True,
    }

    if plan_overrides:
        plan_kwargs.update(plan_overrides)
    if flesh_overrides:
        flesh_kwargs.update(flesh_overrides)
    if write_overrides:
        write_kwargs.update(write_overrides)

    return [
        PhaseConfig(**plan_kwargs),
        PhaseConfig(**flesh_kwargs),
        PhaseConfig(**write_kwargs),
    ]


def create_abridge_run_config(
    *,
    book_id: str,
    book_name: str,
    author_name: str,
    input_file: Path | None = None,
    output_dir: Path | None = None,
    original_file: Path | None = None,
    phases: list[PhaseConfig] | None = None,
    max_workers: int | None = DEFAULT_ABRIDGE_MAX_WORKERS,
    plan_overrides: Mapping[str, Any] | None = None,
    flesh_overrides: Mapping[str, Any] | None = None,
    write_overrides: Mapping[str, Any] | None = None,
    write_config_overrides: Mapping[str, Any] | None = None,
) -> RunConfig:
    """Create a book-specific abridge RunConfig that inherits shared defaults."""
    book_output_dir = output_dir or Path(f"books/{book_id}/output")
    abridge_input_file = input_file or book_output_dir / DEFAULT_ABRIDGE_MODERNIZED_FILENAME

    return RunConfig(
        book_id=book_id,
        book_name=book_name,
        author_name=author_name,
        input_file=abridge_input_file,
        output_dir=book_output_dir,
        original_file=original_file or abridge_input_file,
        phases=phases
        if phases is not None
        else default_abridge_phases(
            plan_overrides=plan_overrides,
            flesh_overrides=flesh_overrides,
            write_overrides=write_overrides,
            write_config_overrides=write_config_overrides,
        ),
        max_workers=max_workers,
    )
