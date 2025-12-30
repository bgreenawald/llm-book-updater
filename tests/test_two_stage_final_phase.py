"""
Tests for the TwoStageFinalPhase and TwoStageModelConfig.

These tests verify the two-stage FINAL phase implementation, including:
- TwoStageModelConfig validation
- TwoStageFinalPhase initialization
- Block processing in both batch and non-batch modes
- Debug JSON file output
- Error handling and retry logic
- Cost tracking for both stages
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from book_updater import PhaseConfig, PhaseType, TwoStageModelConfig
from llm_core import Provider, ModelConfig


class TestTwoStageModelConfig:
    """Tests for TwoStageModelConfig dataclass validation."""

    def test_valid_config_creation(self):
        """Test creating a valid TwoStageModelConfig."""
        identify_model = ModelConfig(
            provider=Provider.OPENROUTER,
            model_id="deepseek/deepseek-r1",
            provider_model_name="deepseek-r1",
        )
        implement_model = ModelConfig(
            provider=Provider.GEMINI,
            model_id="google/gemini-2.5-flash",
            provider_model_name="gemini-2.5-flash",
        )

        config = TwoStageModelConfig(
            identify_model=identify_model,
            implement_model=implement_model,
        )

        assert config.identify_model == identify_model
        assert config.implement_model == implement_model
        assert config.identify_reasoning is None
        assert config.implement_reasoning is None

    def test_config_with_identify_reasoning(self):
        """Test creating config with identify_reasoning set."""
        identify_model = ModelConfig(
            provider=Provider.OPENROUTER,
            model_id="deepseek/deepseek-r1",
            provider_model_name="deepseek-r1",
        )
        implement_model = ModelConfig(
            provider=Provider.GEMINI,
            model_id="google/gemini-2.5-flash",
            provider_model_name="gemini-2.5-flash",
        )

        config = TwoStageModelConfig(
            identify_model=identify_model,
            implement_model=implement_model,
            identify_reasoning={"effort": "high"},
        )

        assert config.identify_reasoning == {"effort": "high"}
        assert config.implement_reasoning is None

    def test_config_with_implement_reasoning(self):
        """Test creating config with implement_reasoning set."""
        identify_model = ModelConfig(
            provider=Provider.OPENROUTER,
            model_id="deepseek/deepseek-r1",
            provider_model_name="deepseek-r1",
        )
        implement_model = ModelConfig(
            provider=Provider.GEMINI,
            model_id="google/gemini-2.5-flash",
            provider_model_name="gemini-2.5-flash",
        )

        config = TwoStageModelConfig(
            identify_model=identify_model,
            implement_model=implement_model,
            implement_reasoning={"effort": "medium"},
        )

        assert config.identify_reasoning is None
        assert config.implement_reasoning == {"effort": "medium"}

    def test_config_with_both_reasoning(self):
        """Test creating config with both identify and implement reasoning set."""
        identify_model = ModelConfig(
            provider=Provider.OPENROUTER,
            model_id="deepseek/deepseek-r1",
            provider_model_name="deepseek-r1",
        )
        implement_model = ModelConfig(
            provider=Provider.GEMINI,
            model_id="google/gemini-2.5-flash",
            provider_model_name="gemini-2.5-flash",
        )

        config = TwoStageModelConfig(
            identify_model=identify_model,
            implement_model=implement_model,
            identify_reasoning={"effort": "high"},
            implement_reasoning={"effort": "low"},
        )

        assert config.identify_reasoning == {"effort": "high"}
        assert config.implement_reasoning == {"effort": "low"}

    def test_invalid_identify_model_type(self):
        """Test that invalid identify_model type raises ValidationError."""
        implement_model = ModelConfig(
            provider=Provider.GEMINI,
            model_id="google/gemini-2.5-flash",
            provider_model_name="gemini-2.5-flash",
        )

        with pytest.raises(ValidationError):
            TwoStageModelConfig(
                identify_model="not_a_model_config",  # type: ignore
                implement_model=implement_model,
            )

    def test_invalid_implement_model_type(self):
        """Test that invalid implement_model type raises ValidationError."""
        identify_model = ModelConfig(
            provider=Provider.OPENROUTER,
            model_id="deepseek/deepseek-r1",
            provider_model_name="deepseek-r1",
        )

        with pytest.raises(ValidationError):
            TwoStageModelConfig(
                identify_model=identify_model,
                implement_model="not_a_model_config",  # type: ignore
            )

    def test_invalid_identify_reasoning_type(self):
        """Test that non-dict identify_reasoning raises ValidationError."""
        identify_model = ModelConfig(
            provider=Provider.OPENROUTER,
            model_id="deepseek/deepseek-r1",
            provider_model_name="deepseek-r1",
        )
        implement_model = ModelConfig(
            provider=Provider.GEMINI,
            model_id="google/gemini-2.5-flash",
            provider_model_name="gemini-2.5-flash",
        )

        with pytest.raises(ValidationError):
            TwoStageModelConfig(
                identify_model=identify_model,
                implement_model=implement_model,
                identify_reasoning="not_a_dict",  # type: ignore
            )

    def test_invalid_implement_reasoning_type(self):
        """Test that non-dict implement_reasoning raises ValidationError."""
        identify_model = ModelConfig(
            provider=Provider.OPENROUTER,
            model_id="deepseek/deepseek-r1",
            provider_model_name="deepseek-r1",
        )
        implement_model = ModelConfig(
            provider=Provider.GEMINI,
            model_id="google/gemini-2.5-flash",
            provider_model_name="gemini-2.5-flash",
        )

        with pytest.raises(ValidationError):
            TwoStageModelConfig(
                identify_model=identify_model,
                implement_model=implement_model,
                implement_reasoning="not_a_dict",  # type: ignore
            )

    def test_invalid_identify_reasoning_key_type(self):
        """Test that non-string keys in identify_reasoning raise ValidationError."""
        identify_model = ModelConfig(
            provider=Provider.OPENROUTER,
            model_id="deepseek/deepseek-r1",
            provider_model_name="deepseek-r1",
        )
        implement_model = ModelConfig(
            provider=Provider.GEMINI,
            model_id="google/gemini-2.5-flash",
            provider_model_name="gemini-2.5-flash",
        )

        with pytest.raises(ValidationError):
            TwoStageModelConfig(
                identify_model=identify_model,
                implement_model=implement_model,
                identify_reasoning={123: "value"},  # type: ignore
            )

    def test_invalid_implement_reasoning_key_type(self):
        """Test that non-string keys in implement_reasoning raise ValidationError."""
        identify_model = ModelConfig(
            provider=Provider.OPENROUTER,
            model_id="deepseek/deepseek-r1",
            provider_model_name="deepseek-r1",
        )
        implement_model = ModelConfig(
            provider=Provider.GEMINI,
            model_id="google/gemini-2.5-flash",
            provider_model_name="gemini-2.5-flash",
        )

        with pytest.raises(ValidationError):
            TwoStageModelConfig(
                identify_model=identify_model,
                implement_model=implement_model,
                implement_reasoning={123: "value"},  # type: ignore
            )


class TestPhaseConfigWithTwoStage:
    """Tests for PhaseConfig validation with FINAL_TWO_STAGE."""

    def test_final_two_stage_requires_two_stage_config(self):
        """Test that FINAL_TWO_STAGE requires two_stage_config."""
        with pytest.raises(ValidationError):
            PhaseConfig(
                phase_type=PhaseType.FINAL_TWO_STAGE,
                enabled=True,
            )

    def test_final_two_stage_with_valid_config(self):
        """Test FINAL_TWO_STAGE with valid two_stage_config."""
        identify_model = ModelConfig(
            provider=Provider.OPENROUTER,
            model_id="deepseek/deepseek-r1",
            provider_model_name="deepseek-r1",
        )
        implement_model = ModelConfig(
            provider=Provider.GEMINI,
            model_id="google/gemini-2.5-flash",
            provider_model_name="gemini-2.5-flash",
        )

        two_stage_config = TwoStageModelConfig(
            identify_model=identify_model,
            implement_model=implement_model,
        )

        phase_config = PhaseConfig(
            phase_type=PhaseType.FINAL_TWO_STAGE,
            enabled=True,
            two_stage_config=two_stage_config,
        )

        assert phase_config.two_stage_config == two_stage_config
        assert phase_config.name == "final_two_stage"

    def test_non_two_stage_rejects_two_stage_config(self):
        """Test that non-FINAL_TWO_STAGE phases reject two_stage_config."""
        identify_model = ModelConfig(
            provider=Provider.OPENROUTER,
            model_id="deepseek/deepseek-r1",
            provider_model_name="deepseek-r1",
        )
        implement_model = ModelConfig(
            provider=Provider.GEMINI,
            model_id="google/gemini-2.5-flash",
            provider_model_name="gemini-2.5-flash",
        )

        two_stage_config = TwoStageModelConfig(
            identify_model=identify_model,
            implement_model=implement_model,
        )

        with pytest.raises(ValidationError):
            PhaseConfig(
                phase_type=PhaseType.FINAL,
                enabled=True,
                two_stage_config=two_stage_config,
            )

    def test_final_two_stage_no_default_prompt_paths(self):
        """Test that FINAL_TWO_STAGE doesn't set default prompt paths."""
        identify_model = ModelConfig(
            provider=Provider.OPENROUTER,
            model_id="deepseek/deepseek-r1",
            provider_model_name="deepseek-r1",
        )
        implement_model = ModelConfig(
            provider=Provider.GEMINI,
            model_id="google/gemini-2.5-flash",
            provider_model_name="gemini-2.5-flash",
        )

        two_stage_config = TwoStageModelConfig(
            identify_model=identify_model,
            implement_model=implement_model,
        )

        phase_config = PhaseConfig(
            phase_type=PhaseType.FINAL_TWO_STAGE,
            enabled=True,
            two_stage_config=two_stage_config,
        )

        # Prompt paths should not be set (handled internally by the phase)
        assert phase_config.system_prompt_path is None
        assert phase_config.user_prompt_path is None


class TestTwoStageFinalPhaseInitialization:
    """Tests for TwoStageFinalPhase initialization."""

    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create input file
            input_file = temp_path / "input.md"
            input_file.write_text("## Chapter 1\n\nThis is the content.\n\n## Chapter 2\n\nMore content here.\n")

            # Create original file
            original_file = temp_path / "original.md"
            original_file.write_text(
                "## Chapter 1\n\nThis is the original content.\n\n## Chapter 2\n\nOriginal more content.\n"
            )

            # Create output file path
            output_file = temp_path / "output.md"

            # Create prompt files
            prompts_dir = temp_path / "prompts"
            prompts_dir.mkdir()

            (prompts_dir / "final_identify_system.md").write_text("IDENTIFY system prompt")
            (prompts_dir / "final_identify_user.md").write_text(
                "IDENTIFY: {current_body} vs {original_body} for {book_name}"
            )
            (prompts_dir / "final_implement_system.md").write_text("IMPLEMENT system prompt")
            (prompts_dir / "final_implement_user.md").write_text("IMPLEMENT: {current_body} with changes: {changes}")

            yield {
                "temp_path": temp_path,
                "input_file": input_file,
                "original_file": original_file,
                "output_file": output_file,
                "prompts_dir": prompts_dir,
            }

    @pytest.fixture
    def mock_models(self):
        """Create mock LLM models."""
        identify_model = MagicMock()
        identify_model.chat_completion.return_value = (
            "### Change 1: Deeper Fidelity\n\n**Location**: First sentence",
            "gen-identify-123",
        )
        identify_model.supports_batch.return_value = False
        identify_model.__str__ = lambda self: "mock-identify-model"

        implement_model = MagicMock()
        implement_model.chat_completion.return_value = ("Refined content here.", "gen-implement-456")
        implement_model.supports_batch.return_value = False
        implement_model.__str__ = lambda self: "mock-implement-model"

        return {"identify": identify_model, "implement": implement_model}

    def test_initialization_success(self, temp_files, mock_models):
        """Test successful initialization of TwoStageFinalPhase."""
        from book_updater.phases import StageConfig, TwoStageFinalPhase

        # Create stage configs
        identify_config = StageConfig(
            model=mock_models["identify"],
            system_prompt="IDENTIFY system prompt",
            user_prompt_template="Process {current_body}",
        )
        implement_config = StageConfig(
            model=mock_models["implement"],
            system_prompt="IMPLEMENT system prompt",
            user_prompt_template="Apply changes: {changes}",
        )

        phase = TwoStageFinalPhase(
            name="final_two_stage",
            input_file_path=temp_files["input_file"],
            output_file_path=temp_files["output_file"],
            original_file_path=temp_files["original_file"],
            book_name="Test Book",
            author_name="Test Author",
            identify_config=identify_config,
            implement_config=implement_config,
        )

        assert phase.name == "final_two_stage"
        assert phase.identify_config.model == mock_models["identify"]
        assert phase.implement_config.model == mock_models["implement"]


class TestTwoStageFinalPhaseProcessing:
    """Tests for TwoStageFinalPhase block processing."""

    @pytest.fixture
    def phase_with_mocks(self):
        """Create a TwoStageFinalPhase with mocked models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files
            input_file = temp_path / "input.md"
            input_file.write_text("## Chapter 1\n\nThis is the content.\n\n")

            original_file = temp_path / "original.md"
            original_file.write_text("## Chapter 1\n\nThis is the original content.\n\n")

            output_file = temp_path / "output.md"

            # Create prompt files
            prompts_dir = temp_path / "prompts"
            prompts_dir.mkdir()
            (prompts_dir / "final_identify_system.md").write_text("IDENTIFY system")
            (prompts_dir / "final_identify_user.md").write_text("{current_body} {original_body} {current_header}")
            (prompts_dir / "final_implement_system.md").write_text("IMPLEMENT system")
            (prompts_dir / "final_implement_user.md").write_text("{current_body} {current_header} {changes}")

            # Create mock models
            identify_model = MagicMock()
            identify_model.chat_completion.return_value = (
                "### No Changes Recommended\n\nThe passage is already excellent.",
                "gen-identify-123",
            )
            identify_model.supports_batch.return_value = False
            identify_model.__str__ = lambda self: "mock-identify"

            implement_model = MagicMock()
            implement_model.chat_completion.return_value = ("This is the refined content.", "gen-implement-456")
            implement_model.supports_batch.return_value = False
            implement_model.__str__ = lambda self: "mock-implement"

            from book_updater.phases import StageConfig, TwoStageFinalPhase

            # Create stage configs
            identify_config = StageConfig(
                model=identify_model,
                system_prompt="IDENTIFY system prompt",
                user_prompt_template="Process {current_body}",
            )
            implement_config = StageConfig(
                model=implement_model,
                system_prompt="IMPLEMENT system prompt",
                user_prompt_template="Apply changes: {changes}",
            )

            phase = TwoStageFinalPhase(
                name="final_two_stage",
                input_file_path=input_file,
                output_file_path=output_file,
                original_file_path=original_file,
                book_name="Test Book",
                author_name="Test Author",
                identify_config=identify_config,
                implement_config=implement_config,
            )

            yield {
                "phase": phase,
                "identify_model": identify_model,
                "implement_model": implement_model,
                "temp_path": temp_path,
            }

    def test_process_block_calls_both_models(self, phase_with_mocks):
        """Test that _process_single_block calls both identify and implement models."""
        phase = phase_with_mocks["phase"]
        identify_model = phase_with_mocks["identify_model"]
        implement_model = phase_with_mocks["implement_model"]

        current_block = "## Chapter 1\n\nThis is the content."
        original_block = "## Chapter 1\n\nThis is the original content."

        with patch("book_updater.phases.two_stage.add_generation_id"):
            result = phase._process_single_block(current_block, original_block, block_index=0)

        # Both models should have been called
        assert identify_model.chat_completion.called
        assert implement_model.chat_completion.called

        # Result should contain the header and processed body
        assert "## Chapter 1" in result
        assert "This is the refined content." in result

    def test_process_block_skips_empty_content(self, phase_with_mocks):
        """Test that empty blocks are skipped."""
        phase = phase_with_mocks["phase"]
        identify_model = phase_with_mocks["identify_model"]

        current_block = "## Empty Chapter\n\n"
        original_block = "## Empty Chapter\n\n"

        result = phase._process_single_block(current_block, original_block, block_index=0)

        # Should not call models for empty content
        assert not identify_model.chat_completion.called

        # Should return the block as-is
        assert "## Empty Chapter" in result

    def test_process_block_skips_special_tags_only(self, phase_with_mocks):
        """Test that blocks with only special tags are skipped."""
        phase = phase_with_mocks["phase"]
        identify_model = phase_with_mocks["identify_model"]

        current_block = "## Preface\n\n{preface}"
        original_block = "## Preface\n\n{preface}"

        result = phase._process_single_block(current_block, original_block, block_index=0)

        # Should not call models for special-tag-only content
        assert not identify_model.chat_completion.called

        # Should return the block as-is with the tag
        assert "{preface}" in result

    def test_debug_data_collected(self, phase_with_mocks):
        """Test that debug data is collected during processing."""
        phase = phase_with_mocks["phase"]

        current_block = "## Chapter 1\n\nThis is the content."
        original_block = "## Chapter 1\n\nThis is the original content."

        with patch("book_updater.phases.two_stage.add_generation_id"):
            phase._process_single_block(current_block, original_block, block_index=0)

        # Debug data should be collected
        assert len(phase._debug_data) == 1
        assert phase._debug_data[0]["block_index"] == 0
        assert phase._debug_data[0]["header"] == "## Chapter 1"
        assert "### No Changes" in phase._debug_data[0]["identify_response"]
        assert phase._debug_data[0]["generation_id"] == "gen-identify-123"


class TestTwoStageFinalPhaseRun:
    """Tests for TwoStageFinalPhase full run."""

    @pytest.fixture
    def full_phase_setup(self):
        """Create a full setup for running the phase."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create input file with multiple blocks
            input_file = temp_path / "input.md"
            input_file.write_text(
                "## Chapter 1\n\nFirst chapter content.\n\n## Chapter 2\n\nSecond chapter content.\n\n"
            )

            original_file = temp_path / "original.md"
            original_file.write_text(
                "## Chapter 1\n\nOriginal first chapter.\n\n## Chapter 2\n\nOriginal second chapter.\n\n"
            )

            output_dir = temp_path / "output"
            output_dir.mkdir()
            output_file = output_dir / "output.md"

            # Create prompt files
            prompts_dir = temp_path / "prompts"
            prompts_dir.mkdir()
            (prompts_dir / "final_identify_system.md").write_text("IDENTIFY system")
            (prompts_dir / "final_identify_user.md").write_text("{current_body} {original_body} {current_header}")
            (prompts_dir / "final_implement_system.md").write_text("IMPLEMENT system")
            (prompts_dir / "final_implement_user.md").write_text("{current_body} {current_header} {changes}")

            # Create mock models
            identify_model = MagicMock()
            identify_model.chat_completion.return_value = ("### No Changes", "gen-id")
            identify_model.supports_batch.return_value = False
            identify_model.__str__ = lambda self: "identify-model"

            implement_model = MagicMock()
            implement_model.chat_completion.return_value = ("Refined.", "gen-impl")
            implement_model.supports_batch.return_value = False
            implement_model.__str__ = lambda self: "implement-model"

            from book_updater.phases import StageConfig, TwoStageFinalPhase

            # Load prompts from files
            identify_system = (prompts_dir / "final_identify_system.md").read_text()
            identify_user = (prompts_dir / "final_identify_user.md").read_text()
            implement_system = (prompts_dir / "final_implement_system.md").read_text()
            implement_user = (prompts_dir / "final_implement_user.md").read_text()

            # Create stage configs
            identify_config = StageConfig(
                model=identify_model,
                system_prompt=identify_system,
                user_prompt_template=identify_user,
            )
            implement_config = StageConfig(
                model=implement_model,
                system_prompt=implement_system,
                user_prompt_template=implement_user,
            )

            phase = TwoStageFinalPhase(
                name="final_two_stage",
                input_file_path=input_file,
                output_file_path=output_file,
                original_file_path=original_file,
                book_name="Test Book",
                author_name="Test Author",
                identify_config=identify_config,
                implement_config=implement_config,
                max_workers=1,
            )

            yield {
                "phase": phase,
                "output_dir": output_dir,
                "output_file": output_file,
            }

    def test_run_creates_output_file(self, full_phase_setup):
        """Test that run() creates the output file."""
        phase = full_phase_setup["phase"]
        output_file = full_phase_setup["output_file"]

        with patch("book_updater.phases.two_stage.add_generation_id"):
            phase.run()

        assert output_file.exists()
        content = output_file.read_text()
        assert "## Chapter 1" in content
        assert "## Chapter 2" in content

    def test_run_creates_debug_file(self, full_phase_setup):
        """Test that run() creates the debug JSON file."""
        phase = full_phase_setup["phase"]
        output_dir = full_phase_setup["output_dir"]

        with patch("book_updater.phases.two_stage.add_generation_id"):
            phase.run()

        # Find the debug file
        debug_files = list(output_dir.glob("final_identify_debug_*.json"))
        assert len(debug_files) == 1

        # Verify debug file content
        with open(debug_files[0]) as f:
            debug_data = json.load(f)

        assert debug_data["phase_name"] == "final_two_stage"
        assert debug_data["book_name"] == "Test Book"
        assert debug_data["author_name"] == "Test Author"
        assert len(debug_data["blocks"]) == 2

    def test_debug_file_structure(self, full_phase_setup):
        """Test the structure of the debug JSON file."""
        phase = full_phase_setup["phase"]
        output_dir = full_phase_setup["output_dir"]

        with patch("book_updater.phases.two_stage.add_generation_id"):
            phase.run()

        debug_files = list(output_dir.glob("final_identify_debug_*.json"))
        with open(debug_files[0]) as f:
            debug_data = json.load(f)

        # Check required fields
        assert "timestamp" in debug_data
        assert "identify_model" in debug_data
        assert "implement_model" in debug_data

        # Check block structure
        for block in debug_data["blocks"]:
            assert "block_index" in block
            assert "header" in block
            assert "identify_response" in block
            assert "generation_id" in block


class TestTwoStageFinalPhaseBatchMode:
    """Tests for TwoStageFinalPhase batch mode processing."""

    @pytest.fixture
    def batch_phase_setup(self):
        """Create a setup for batch mode testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create input file
            input_file = temp_path / "input.md"
            input_file.write_text("## Chapter 1\n\nContent 1.\n\n## Chapter 2\n\nContent 2.\n\n")

            original_file = temp_path / "original.md"
            original_file.write_text("## Chapter 1\n\nOriginal 1.\n\n## Chapter 2\n\nOriginal 2.\n\n")

            output_dir = temp_path / "output"
            output_dir.mkdir()
            output_file = output_dir / "output.md"

            # Create prompt files
            prompts_dir = temp_path / "prompts"
            prompts_dir.mkdir()
            (prompts_dir / "final_identify_system.md").write_text("ID sys")
            (prompts_dir / "final_identify_user.md").write_text("{current_body}")
            (prompts_dir / "final_implement_system.md").write_text("IMPL sys")
            (prompts_dir / "final_implement_user.md").write_text("{current_body} {changes}")

            yield {
                "temp_path": temp_path,
                "input_file": input_file,
                "original_file": original_file,
                "output_file": output_file,
                "output_dir": output_dir,
                "prompts_dir": prompts_dir,
            }

    def test_batch_mode_activated_when_both_models_support(self, batch_phase_setup):
        """Test that batch mode is activated when both models support it."""
        # Create models that support batch
        identify_model = MagicMock()
        identify_model.supports_batch.return_value = True
        identify_model.batch_chat_completion.return_value = [
            {"content": "Changes 1", "generation_id": "id-1", "metadata": {"index": 0}},
            {"content": "Changes 2", "generation_id": "id-2", "metadata": {"index": 1}},
        ]
        identify_model.__str__ = lambda self: "id-model"

        implement_model = MagicMock()
        implement_model.supports_batch.return_value = True
        implement_model.batch_chat_completion.return_value = [
            {"content": "Refined 1", "generation_id": "impl-1", "metadata": {"index": 0}},
            {"content": "Refined 2", "generation_id": "impl-2", "metadata": {"index": 1}},
        ]
        implement_model.__str__ = lambda self: "impl-model"

        from book_updater.phases import StageConfig, TwoStageFinalPhase

        # Load prompts from files
        prompts_dir = batch_phase_setup["prompts_dir"]
        identify_system = (prompts_dir / "final_identify_system.md").read_text()
        identify_user = (prompts_dir / "final_identify_user.md").read_text()
        implement_system = (prompts_dir / "final_implement_system.md").read_text()
        implement_user = (prompts_dir / "final_implement_user.md").read_text()

        # Create stage configs
        identify_config = StageConfig(
            model=identify_model,
            system_prompt=identify_system,
            user_prompt_template=identify_user,
        )
        implement_config = StageConfig(
            model=implement_model,
            system_prompt=implement_system,
            user_prompt_template=implement_user,
        )

        phase = TwoStageFinalPhase(
            name="batch_test",
            input_file_path=batch_phase_setup["input_file"],
            output_file_path=batch_phase_setup["output_file"],
            original_file_path=batch_phase_setup["original_file"],
            book_name="Test",
            author_name="Author",
            identify_config=identify_config,
            implement_config=implement_config,
            use_batch=True,
        )

        with patch("book_updater.phases.two_stage.add_generation_id"):
            phase.run()

        # Both batch APIs should have been called
        assert identify_model.batch_chat_completion.called
        assert implement_model.batch_chat_completion.called

    def test_fallback_to_non_batch_when_one_model_unsupported(self, batch_phase_setup):
        """Test fallback to non-batch when one model doesn't support batch."""
        # Identify supports batch, implement doesn't
        identify_model = MagicMock()
        identify_model.supports_batch.return_value = True
        identify_model.chat_completion.return_value = ("Changes", "id-1")
        identify_model.__str__ = lambda self: "id-model"

        implement_model = MagicMock()
        implement_model.supports_batch.return_value = False
        implement_model.chat_completion.return_value = ("Refined", "impl-1")
        implement_model.__str__ = lambda self: "impl-model"

        from book_updater.phases import StageConfig, TwoStageFinalPhase

        # Load prompts from files
        prompts_dir = batch_phase_setup["prompts_dir"]
        identify_system = (prompts_dir / "final_identify_system.md").read_text()
        identify_user = (prompts_dir / "final_identify_user.md").read_text()
        implement_system = (prompts_dir / "final_implement_system.md").read_text()
        implement_user = (prompts_dir / "final_implement_user.md").read_text()

        # Create stage configs
        identify_config = StageConfig(
            model=identify_model,
            system_prompt=identify_system,
            user_prompt_template=identify_user,
        )
        implement_config = StageConfig(
            model=implement_model,
            system_prompt=implement_system,
            user_prompt_template=implement_user,
        )

        phase = TwoStageFinalPhase(
            name="mixed_test",
            input_file_path=batch_phase_setup["input_file"],
            output_file_path=batch_phase_setup["output_file"],
            original_file_path=batch_phase_setup["original_file"],
            book_name="Test",
            author_name="Author",
            identify_config=identify_config,
            implement_config=implement_config,
            use_batch=True,
            max_workers=1,
        )

        with patch("book_updater.phases.two_stage.add_generation_id"):
            phase.run()

        # Should fall back to individual calls
        assert identify_model.chat_completion.called
        assert implement_model.chat_completion.called
        # Batch API should not have been called
        assert not identify_model.batch_chat_completion.called

    def test_batch_retry_with_out_of_order_responses(self, batch_phase_setup):
        """Test that batch retry uses correct request when responses are out of order."""
        # Create input file with 3 blocks to test out-of-order handling
        input_file = batch_phase_setup["input_file"]
        input_file.write_text(
            "## Chapter 1\n\nContent 1.\n\n## Chapter 2\n\nContent 2.\n\n## Chapter 3\n\nContent 3.\n\n"
        )

        original_file = batch_phase_setup["original_file"]
        original_file.write_text(
            "## Chapter 1\n\nOriginal 1.\n\n## Chapter 2\n\nOriginal 2.\n\n## Chapter 3\n\nOriginal 3.\n\n"
        )

        # Create models that support batch
        identify_model = MagicMock()
        identify_model.supports_batch.return_value = True
        # Return responses OUT OF ORDER: index 1, then 0 (FAILED), then 2
        identify_model.batch_chat_completion.return_value = [
            {"content": "Changes 2", "generation_id": "id-2", "metadata": {"index": 1}},
            {"content": "Error occurred", "generation_id": "id-error", "metadata": {"index": 0}, "failed": True},
            {"content": "Changes 3", "generation_id": "id-3", "metadata": {"index": 2}},
        ]
        identify_model.__str__ = lambda self: "id-model"

        implement_model = MagicMock()
        implement_model.supports_batch.return_value = True
        # Implement stage responses in order (after identify is fixed)
        implement_model.batch_chat_completion.return_value = [
            {"content": "Refined 1", "generation_id": "impl-1", "metadata": {"index": 0}},
            {"content": "Refined 2", "generation_id": "impl-2", "metadata": {"index": 1}},
            {"content": "Refined 3", "generation_id": "impl-3", "metadata": {"index": 2}},
        ]
        implement_model.__str__ = lambda self: "impl-model"

        from book_updater.phases import StageConfig, TwoStageFinalPhase

        # Load prompts from files
        prompts_dir = batch_phase_setup["prompts_dir"]
        identify_system = (prompts_dir / "final_identify_system.md").read_text()
        identify_user = (prompts_dir / "final_identify_user.md").read_text()
        implement_system = (prompts_dir / "final_implement_system.md").read_text()
        implement_user = (prompts_dir / "final_implement_user.md").read_text()

        # Create stage configs
        identify_config = StageConfig(
            model=identify_model,
            system_prompt=identify_system,
            user_prompt_template=identify_user,
        )
        implement_config = StageConfig(
            model=implement_model,
            system_prompt=implement_system,
            user_prompt_template=implement_user,
        )

        phase = TwoStageFinalPhase(
            name="batch_retry_test",
            input_file_path=batch_phase_setup["input_file"],
            output_file_path=batch_phase_setup["output_file"],
            original_file_path=batch_phase_setup["original_file"],
            book_name="Test",
            author_name="Author",
            identify_config=identify_config,
            implement_config=implement_config,
            use_batch=True,
            enable_retry=True,
        )

        # Track retry calls to verify correct prompt is used
        retry_calls = []

        def mock_retry(*args, **kwargs):
            """Mock make_llm_call_with_retry to capture calls."""
            retry_calls.append(
                {
                    "user_prompt": kwargs.get("user_prompt"),
                    "system_prompt": kwargs.get("system_prompt"),
                    "block_info": kwargs.get("block_info"),
                }
            )
            # Return successful retry result
            return ("Changes 1 (retried)", "id-1-retried")

        with (
            patch("book_updater.phases.two_stage.add_generation_id"),
            patch("book_updater.phases.two_stage.make_llm_call_with_retry", side_effect=mock_retry),
        ):
            phase.run()

        # Verify retry was called exactly once (for the failed response)
        assert len(retry_calls) == 1

        # Verify retry was called with the correct prompt for block index 0
        retry_call = retry_calls[0]
        assert "identify block index 0" in retry_call["block_info"]
        # The user prompt should contain "Content 1" (from block 0), not "Content 2" (from block 1)
        assert "Content 1" in retry_call["user_prompt"]
        assert "Content 2" not in retry_call["user_prompt"]

        # Verify batch API was called
        assert identify_model.batch_chat_completion.called
        assert implement_model.batch_chat_completion.called

        # Verify final output contains all three chapters
        output_content = batch_phase_setup["output_file"].read_text()
        assert "## Chapter 1" in output_content
        assert "## Chapter 2" in output_content
        assert "## Chapter 3" in output_content


class TestPhaseFactoryTwoStage:
    """Tests for PhaseFactory.create_two_stage_final_phase."""

    def test_factory_creates_phase(self):
        """Test that the factory creates a valid TwoStageFinalPhase."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create required files
            input_file = temp_path / "input.md"
            input_file.write_text("## Test\n\nContent.\n\n")
            original_file = temp_path / "original.md"
            original_file.write_text("## Test\n\nOriginal.\n\n")
            output_file = temp_path / "output.md"

            # Create two-stage config
            identify_model_config = ModelConfig(
                provider=Provider.OPENROUTER,
                model_id="test/identify",
                provider_model_name="identify",
            )
            implement_model_config = ModelConfig(
                provider=Provider.OPENROUTER,
                model_id="test/implement",
                provider_model_name="implement",
            )

            two_stage_config = TwoStageModelConfig(
                identify_model=identify_model_config,
                implement_model=implement_model_config,
            )

            # Create phase config
            phase_config = PhaseConfig(
                phase_type=PhaseType.FINAL_TWO_STAGE,
                enabled=True,
                two_stage_config=two_stage_config,
                name="final_two_stage",
                input_file_path=input_file,
                output_file_path=output_file,
                original_file_path=original_file,
                book_name="Test Book",
                author_name="Test Author",
            )

            # Create mock models
            identify_model = MagicMock()
            identify_model.chat_completion.return_value = ("Changes", "id-1")
            identify_model.supports_batch.return_value = False
            identify_model.__str__ = lambda self: "id-model"

            implement_model = MagicMock()
            implement_model.chat_completion.return_value = ("Refined", "impl-1")
            implement_model.supports_batch.return_value = False
            implement_model.__str__ = lambda self: "impl-model"

            from book_updater.phases import PhaseFactory

            phase = PhaseFactory.create_two_stage_final_phase(
                config=phase_config,
                identify_model=identify_model,
                implement_model=implement_model,
            )

            assert phase.name == "final_two_stage"
            assert phase.identify_config.model == identify_model
            assert phase.implement_config.model == implement_model

    def test_factory_requires_two_stage_config(self):
        """Test that factory raises error when two_stage_config is missing."""
        # Create phase config without two_stage_config
        # This should fail at PhaseConfig creation, not factory
        with pytest.raises(ValidationError):
            PhaseConfig(
                phase_type=PhaseType.FINAL_TWO_STAGE,
                enabled=True,
            )
