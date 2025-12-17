from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from src.common.provider import Provider
from src.config import PhaseConfig, PhaseType, RunConfig
from src.llm_model import LlmModel, ModelConfig
from src.pipeline import Pipeline


def test_pipeline_initialization_preserves_llm_kwargs(tmp_path: Path, monkeypatch) -> None:
    """Ensure PhaseConfig.llm_kwargs is preserved when Pipeline builds phase instances.

    This is a regression test: Pipeline used to create a "factory_config" copy of the
    phase config but accidentally dropped llm_kwargs, causing provider parameters to
    never reach LlmPhase/LLM calls.
    """
    input_path = tmp_path / "input.md"
    input_path.write_text("# Title\n\nSome content.\n", encoding="utf-8")

    output_dir = tmp_path / "out"

    provider_kwargs = {"provider": {"only": ["moonshotai/int4", "parasail/int4"]}}

    config = RunConfig(
        book_id="test_book",
        book_name="Test Book",
        author_name="Test Author",
        input_file=input_path,
        output_dir=output_dir,
        original_file=input_path,
        phases=[
            PhaseConfig(
                phase_type=PhaseType.MODERNIZE,
                model=ModelConfig(provider=Provider.OPENROUTER, model_id="moonshotai/kimi-k2-thinking"),
                llm_kwargs=provider_kwargs,
            )
        ],
    )

    pipeline = Pipeline(config=config)

    mock_model = Mock(spec=LlmModel)
    monkeypatch.setattr(pipeline, "_get_or_create_model", lambda model_config, temperature: mock_model)

    phase = pipeline._initialize_phase(phase_index=0)
    assert phase is not None
    assert phase.llm_kwargs == provider_kwargs
