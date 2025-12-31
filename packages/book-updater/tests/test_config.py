"""Tests for book_updater config module."""

from pathlib import Path

import pytest
from book_updater.config import (
    PhaseConfig,
    PhaseType,
    PostProcessorType,
    RunConfig,
    TwoStageModelConfig,
)
from llm_core import ModelConfig
from llm_core.providers import Provider
from pydantic import ValidationError


class TestPhaseType:
    """Tests for PhaseType enum."""

    def test_all_phase_types_exist(self):
        """Test that all expected phase types exist."""
        assert PhaseType.MODERNIZE is not None
        assert PhaseType.EDIT is not None
        assert PhaseType.ANNOTATE is not None
        assert PhaseType.FINAL is not None
        assert PhaseType.FINAL_TWO_STAGE is not None
        assert PhaseType.INTRODUCTION is not None
        assert PhaseType.SUMMARY is not None

    def test_phase_types_are_unique(self):
        """Test that all phase types have unique values."""
        values = [pt.value for pt in PhaseType]
        assert len(values) == len(set(values))


class TestPostProcessorType:
    """Tests for PostProcessorType enum."""

    def test_basic_formatting_processors_exist(self):
        """Test basic formatting processors exist."""
        assert PostProcessorType.ENSURE_BLANK_LINE is not None
        assert PostProcessorType.REMOVE_TRAILING_WHITESPACE is not None
        assert PostProcessorType.REMOVE_XML_TAGS is not None
        assert PostProcessorType.REMOVE_MARKDOWN_BLOCKS is not None

    def test_content_preservation_processors_exist(self):
        """Test content preservation processors exist."""
        assert PostProcessorType.NO_NEW_HEADERS is not None
        assert PostProcessorType.REVERT_REMOVED_BLOCK_LINES is not None
        assert PostProcessorType.PRESERVE_F_STRING_TAGS is not None

    def test_validation_processors_exist(self):
        """Test validation processors exist."""
        assert PostProcessorType.VALIDATE_NON_EMPTY_SECTION is not None


class TestPhaseConfig:
    """Tests for PhaseConfig."""

    def test_minimal_config(self):
        """Test creating config with minimal required fields."""
        config = PhaseConfig(phase_type=PhaseType.MODERNIZE)
        assert config.phase_type == PhaseType.MODERNIZE
        assert config.enabled is True
        assert config.name == "modernize"

    def test_default_prompt_paths(self):
        """Test default prompt paths are set correctly."""
        config = PhaseConfig(phase_type=PhaseType.EDIT)
        assert config.system_prompt_path == Path("./prompts/edit_system.md")
        assert config.user_prompt_path == Path("./prompts/edit_user.md")

    def test_disabled_phase(self):
        """Test creating a disabled phase."""
        config = PhaseConfig(phase_type=PhaseType.ANNOTATE, enabled=False)
        assert config.enabled is False

    def test_custom_name(self):
        """Test setting custom name."""
        config = PhaseConfig(phase_type=PhaseType.MODERNIZE, name="my_modernize")
        assert config.name == "my_modernize"

    def test_batch_size_without_use_batch_raises(self):
        """Test that batch_size without use_batch raises error."""
        with pytest.raises(ValidationError) as exc_info:
            PhaseConfig(phase_type=PhaseType.MODERNIZE, batch_size=10, use_batch=False)
        assert "use_batch" in str(exc_info.value)

    def test_batch_size_with_use_batch_valid(self):
        """Test that batch_size with use_batch is valid."""
        config = PhaseConfig(phase_type=PhaseType.MODERNIZE, batch_size=10, use_batch=True)
        assert config.batch_size == 10
        assert config.use_batch is True

    def test_max_subblock_tokens_less_than_min_raises(self):
        """Test that max_subblock_tokens <= min_subblock_tokens raises error."""
        with pytest.raises(ValidationError) as exc_info:
            PhaseConfig(
                phase_type=PhaseType.MODERNIZE,
                max_subblock_tokens=100,
                min_subblock_tokens=200,
            )
        assert "max_subblock_tokens" in str(exc_info.value)

    def test_reasoning_validation_valid_dict(self):
        """Test reasoning validation accepts valid dict."""
        config = PhaseConfig(
            phase_type=PhaseType.MODERNIZE,
            reasoning={"thinking_budget": "1000"},
        )
        assert config.reasoning == {"thinking_budget": "1000"}

    def test_reasoning_validation_none_valid(self):
        """Test reasoning validation accepts None."""
        config = PhaseConfig(phase_type=PhaseType.MODERNIZE, reasoning=None)
        assert config.reasoning is None

    def test_final_two_stage_requires_config(self):
        """Test FINAL_TWO_STAGE requires two_stage_config."""
        with pytest.raises(ValidationError) as exc_info:
            PhaseConfig(phase_type=PhaseType.FINAL_TWO_STAGE)
        assert "two_stage_config" in str(exc_info.value)

    def test_two_stage_config_only_for_final_two_stage(self):
        """Test two_stage_config is only valid for FINAL_TWO_STAGE."""
        two_stage = TwoStageModelConfig(
            identify_model=ModelConfig(provider=Provider.GEMINI, model_id="test1"),
            implement_model=ModelConfig(provider=Provider.GEMINI, model_id="test2"),
        )
        with pytest.raises(ValidationError) as exc_info:
            PhaseConfig(phase_type=PhaseType.MODERNIZE, two_stage_config=two_stage)
        assert "only valid for FINAL_TWO_STAGE" in str(exc_info.value)


class TestTwoStageModelConfig:
    """Tests for TwoStageModelConfig."""

    def test_minimal_config(self):
        """Test creating config with required fields."""
        config = TwoStageModelConfig(
            identify_model=ModelConfig(provider=Provider.GEMINI, model_id="test1"),
            implement_model=ModelConfig(provider=Provider.GEMINI, model_id="test2"),
        )
        assert config.identify_model.model_id == "test1"
        assert config.implement_model.model_id == "test2"
        assert config.identify_reasoning is None
        assert config.implement_reasoning is None

    def test_with_reasoning(self):
        """Test config with reasoning parameters."""
        config = TwoStageModelConfig(
            identify_model=ModelConfig(provider=Provider.GEMINI, model_id="test1"),
            implement_model=ModelConfig(provider=Provider.GEMINI, model_id="test2"),
            identify_reasoning={"budget": "1000"},
            implement_reasoning={"budget": "2000"},
        )
        assert config.identify_reasoning == {"budget": "1000"}
        assert config.implement_reasoning == {"budget": "2000"}


class TestRunConfig:
    """Tests for RunConfig."""

    def test_minimal_config(self, temp_dir, sample_input_file, sample_output_dir):
        """Test creating config with minimal required fields."""
        config = RunConfig(
            book_id="test_book",
            book_name="Test Book",
            author_name="Test Author",
            input_file=sample_input_file,
            output_dir=sample_output_dir,
            original_file=sample_input_file,
            phases=[],
        )
        assert config.book_id == "test_book"
        assert config.book_name == "Test Book"
        assert config.phases == []

    def test_default_values(self, temp_dir, sample_input_file, sample_output_dir):
        """Test default values are set correctly."""
        config = RunConfig(
            book_id="test_book",
            book_name="Test Book",
            author_name="Test Author",
            input_file=sample_input_file,
            output_dir=sample_output_dir,
            original_file=sample_input_file,
        )
        assert config.start_from_phase == 0
        assert config.max_workers is None
        assert len(config.tags_to_preserve) > 0  # Has default tags

    def test_with_phases(self, temp_dir, sample_input_file, sample_output_dir):
        """Test config with phases."""
        phases = [
            PhaseConfig(phase_type=PhaseType.MODERNIZE),
            PhaseConfig(phase_type=PhaseType.EDIT),
        ]
        config = RunConfig(
            book_id="test_book",
            book_name="Test Book",
            author_name="Test Author",
            input_file=sample_input_file,
            output_dir=sample_output_dir,
            original_file=sample_input_file,
            phases=phases,
        )
        assert len(config.phases) == 2
        assert config.phases[0].phase_type == PhaseType.MODERNIZE
        assert config.phases[1].phase_type == PhaseType.EDIT

    def test_start_from_phase_out_of_range_raises(self, temp_dir, sample_input_file, sample_output_dir):
        """Test start_from_phase out of range raises error."""
        phases = [PhaseConfig(phase_type=PhaseType.MODERNIZE)]
        with pytest.raises(ValidationError) as exc_info:
            RunConfig(
                book_id="test_book",
                book_name="Test Book",
                author_name="Test Author",
                input_file=sample_input_file,
                output_dir=sample_output_dir,
                original_file=sample_input_file,
                phases=phases,
                start_from_phase=5,
            )
        assert "start_from_phase" in str(exc_info.value)

    def test_get_phase_order(self, temp_dir, sample_input_file, sample_output_dir):
        """Test get_phase_order returns correct order."""
        phases = [
            PhaseConfig(phase_type=PhaseType.MODERNIZE),
            PhaseConfig(phase_type=PhaseType.EDIT),
            PhaseConfig(phase_type=PhaseType.FINAL),
        ]
        config = RunConfig(
            book_id="test_book",
            book_name="Test Book",
            author_name="Test Author",
            input_file=sample_input_file,
            output_dir=sample_output_dir,
            original_file=sample_input_file,
            phases=phases,
        )
        order = config.get_phase_order()
        assert order == [PhaseType.MODERNIZE, PhaseType.EDIT, PhaseType.FINAL]

    def test_creates_output_directory(self, temp_dir, sample_input_file):
        """Test that output directory is created if it doesn't exist."""
        output_dir = temp_dir / "new_output"
        assert not output_dir.exists()

        RunConfig(
            book_id="test_book",
            book_name="Test Book",
            author_name="Test Author",
            input_file=sample_input_file,
            output_dir=output_dir,
            original_file=sample_input_file,
            phases=[],
        )

        assert output_dir.exists()
