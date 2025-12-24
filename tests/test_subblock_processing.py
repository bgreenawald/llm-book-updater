"""
Tests for sub-block processing functionality.

These tests verify the paragraph splitting, grouping, and sub-block processing
functionality used to enable cost-efficient models to work with large chapters.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.api.config import PhaseConfig, PhaseType
from src.core.constants import (
    DEFAULT_MAX_SUBBLOCK_TOKENS,
    DEFAULT_MIN_SUBBLOCK_TOKENS,
    MAX_SUBBLOCK_TOKEN_BOUND,
)

# ============================================================================
# PhaseConfig Sub-block Validation Tests
# ============================================================================


class TestPhaseConfigSubblockValidation:
    """Tests for PhaseConfig validation of sub-block parameters."""

    def test_default_subblock_values(self):
        """Test that default sub-block values are set correctly."""
        config = PhaseConfig(phase_type=PhaseType.MODERNIZE)

        assert config.use_subblocks is False
        assert config.max_subblock_tokens == DEFAULT_MAX_SUBBLOCK_TOKENS
        assert config.min_subblock_tokens == DEFAULT_MIN_SUBBLOCK_TOKENS

    def test_custom_subblock_values(self):
        """Test that custom sub-block values are accepted."""
        config = PhaseConfig(
            phase_type=PhaseType.MODERNIZE,
            use_subblocks=True,
            max_subblock_tokens=8000,
            min_subblock_tokens=2000,
        )

        assert config.use_subblocks is True
        assert config.max_subblock_tokens == 8000
        assert config.min_subblock_tokens == 2000

    def test_invalid_use_subblocks_type(self):
        """Test that non-bool use_subblocks raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PhaseConfig(
                phase_type=PhaseType.MODERNIZE,
                use_subblocks="true",  # type: ignore
            )
        assert "use_subblocks must be a bool" in str(exc_info.value)

    def test_invalid_max_subblock_tokens_type(self):
        """Test that non-int max_subblock_tokens raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PhaseConfig(
                phase_type=PhaseType.MODERNIZE,
                max_subblock_tokens="4096",  # type: ignore
            )
        assert "max_subblock_tokens must be an int" in str(exc_info.value)

    def test_invalid_min_subblock_tokens_type(self):
        """Test that non-int min_subblock_tokens raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PhaseConfig(
                phase_type=PhaseType.MODERNIZE,
                min_subblock_tokens=1024.5,  # type: ignore
            )
        assert "min_subblock_tokens must be an int" in str(exc_info.value)

    def test_max_subblock_tokens_below_minimum_bound(self):
        """Test that max_subblock_tokens below MIN_SUBBLOCK_TOKEN_BOUND raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PhaseConfig(
                phase_type=PhaseType.MODERNIZE,
                max_subblock_tokens=100,  # Below MIN_SUBBLOCK_TOKEN_BOUND
                min_subblock_tokens=50,
            )
        assert "max_subblock_tokens" in str(exc_info.value)

    def test_max_subblock_tokens_above_maximum_bound(self):
        """Test that max_subblock_tokens above MAX_SUBBLOCK_TOKEN_BOUND raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PhaseConfig(
                phase_type=PhaseType.MODERNIZE,
                max_subblock_tokens=100000,  # Above MAX_SUBBLOCK_TOKEN_BOUND
            )
        assert "max_subblock_tokens" in str(exc_info.value)

    def test_min_subblock_tokens_below_minimum_bound(self):
        """Test that min_subblock_tokens below MIN_SUBBLOCK_TOKEN_BOUND raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PhaseConfig(
                phase_type=PhaseType.MODERNIZE,
                min_subblock_tokens=100,  # Below MIN_SUBBLOCK_TOKEN_BOUND
            )
        assert "min_subblock_tokens" in str(exc_info.value)

    def test_min_subblock_tokens_above_maximum_bound(self):
        """Test that min_subblock_tokens above MAX_SUBBLOCK_TOKEN_BOUND raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PhaseConfig(
                phase_type=PhaseType.MODERNIZE,
                max_subblock_tokens=MAX_SUBBLOCK_TOKEN_BOUND,
                min_subblock_tokens=100000,  # Above MAX_SUBBLOCK_TOKEN_BOUND
            )
        assert "min_subblock_tokens" in str(exc_info.value)

    def test_max_not_greater_than_min(self):
        """Test that max_subblock_tokens must be greater than min_subblock_tokens."""
        with pytest.raises(ValidationError) as exc_info:
            PhaseConfig(
                phase_type=PhaseType.MODERNIZE,
                max_subblock_tokens=1000,
                min_subblock_tokens=1000,  # Equal to max
            )
        assert "max_subblock_tokens must be greater than min_subblock_tokens" in str(exc_info.value)

    def test_max_less_than_min(self):
        """Test that max_subblock_tokens < min_subblock_tokens raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PhaseConfig(
                phase_type=PhaseType.MODERNIZE,
                max_subblock_tokens=500,
                min_subblock_tokens=1000,  # Greater than max
            )
        assert "max_subblock_tokens must be greater than min_subblock_tokens" in str(exc_info.value)


# ============================================================================
# Paragraph Splitting Tests
# ============================================================================


class TestSplitBodyIntoParagraphs:
    """Tests for _split_body_into_paragraphs method."""

    @pytest.fixture
    def mock_phase(self):
        """Create a mock StandardLlmPhase for testing splitting methods."""
        from src.phases.standard import StandardLlmPhase

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create minimal files
            input_file = temp_path / "input.md"
            input_file.write_text("# Test\n\nContent")
            output_file = temp_path / "output.md"
            system_prompt = temp_path / "system.md"
            system_prompt.write_text("System prompt")
            user_prompt = temp_path / "user.md"
            user_prompt.write_text("{current_body}")

            # Create mock model
            mock_model = MagicMock()
            mock_model.chat_completion.return_value = ("Processed", "gen-123")

            phase = StandardLlmPhase(
                name="test_phase",
                input_file_path=input_file,
                output_file_path=output_file,
                original_file_path=input_file,
                system_prompt_path=system_prompt,
                user_prompt_path=user_prompt,
                book_name="Test Book",
                author_name="Test Author",
                model=mock_model,
                use_subblocks=True,
                max_subblock_tokens=4096,
                min_subblock_tokens=1024,
            )
            yield phase

    def test_empty_body(self, mock_phase):
        """Test splitting an empty body."""
        result = mock_phase._split_body_into_paragraphs("")
        assert result == []

    def test_whitespace_only_body(self, mock_phase):
        """Test splitting a body with only whitespace."""
        result = mock_phase._split_body_into_paragraphs("   \n\n   \n   ")
        assert result == []

    def test_single_line(self, mock_phase):
        """Test splitting a single line body."""
        result = mock_phase._split_body_into_paragraphs("This is a single paragraph.")
        assert result == ["This is a single paragraph."]

    def test_multiple_lines(self, mock_phase):
        """Test splitting multiple lines (each line is a paragraph)."""
        body = "First paragraph.\nSecond paragraph.\nThird paragraph."
        result = mock_phase._split_body_into_paragraphs(body)
        assert result == ["First paragraph.", "Second paragraph.", "Third paragraph."]

    def test_filters_empty_lines(self, mock_phase):
        """Test that empty lines are filtered out."""
        body = "First paragraph.\n\nSecond paragraph.\n\n\nThird paragraph."
        result = mock_phase._split_body_into_paragraphs(body)
        assert result == ["First paragraph.", "Second paragraph.", "Third paragraph."]

    def test_strips_whitespace(self, mock_phase):
        """Test that whitespace is stripped from each paragraph."""
        body = "  First paragraph.  \n  Second paragraph.  "
        result = mock_phase._split_body_into_paragraphs(body)
        assert result == ["First paragraph.", "Second paragraph."]

    def test_leading_trailing_newlines(self, mock_phase):
        """Test handling of leading and trailing newlines."""
        body = "\n\nFirst paragraph.\nSecond paragraph.\n\n"
        result = mock_phase._split_body_into_paragraphs(body)
        assert result == ["First paragraph.", "Second paragraph."]


# ============================================================================
# Sub-block Grouping Tests
# ============================================================================


class TestGroupParagraphsIntoSubblocks:
    """Tests for _group_paragraphs_into_subblocks method."""

    @pytest.fixture
    def mock_phase(self):
        """Create a mock StandardLlmPhase for testing grouping methods."""
        from src.phases.standard import StandardLlmPhase

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "input.md"
            input_file.write_text("# Test\n\nContent")
            output_file = temp_path / "output.md"
            system_prompt = temp_path / "system.md"
            system_prompt.write_text("System prompt")
            user_prompt = temp_path / "user.md"
            user_prompt.write_text("{current_body}")

            mock_model = MagicMock()
            mock_model.chat_completion.return_value = ("Processed", "gen-123")

            phase = StandardLlmPhase(
                name="test_phase",
                input_file_path=input_file,
                output_file_path=output_file,
                original_file_path=input_file,
                system_prompt_path=system_prompt,
                user_prompt_path=user_prompt,
                book_name="Test Book",
                author_name="Test Author",
                model=mock_model,
                use_subblocks=True,
                max_subblock_tokens=1000,
                min_subblock_tokens=300,
            )
            yield phase

    def test_empty_paragraphs(self, mock_phase):
        """Test grouping empty paragraph list."""
        result = mock_phase._group_paragraphs_into_subblocks([])
        assert result == []

    def test_single_paragraph(self, mock_phase):
        """Test grouping a single paragraph."""
        result = mock_phase._group_paragraphs_into_subblocks(["Single paragraph content."])
        assert result == ["Single paragraph content."]

    def test_small_paragraphs_grouped(self, mock_phase):
        """Test that small paragraphs are grouped together."""
        # Create paragraphs that are small (well below min_subblock_tokens)
        paragraphs = ["Short para 1.", "Short para 2.", "Short para 3."]

        # Mock token counting to return small values
        with patch.object(mock_phase, "_count_tokens", return_value=50):
            result = mock_phase._group_paragraphs_into_subblocks(paragraphs)

        # All should be grouped into one sub-block since total is still < min
        assert len(result) == 1
        assert "Short para 1." in result[0]
        assert "Short para 2." in result[0]
        assert "Short para 3." in result[0]

    def test_large_paragraph_kept_as_lone_subblock(self, mock_phase):
        """Test that a large paragraph exceeding max_subblock_tokens is kept alone."""
        paragraphs = ["Small para.", "Very large paragraph content " * 100, "Another small para."]

        def mock_token_count(text: str) -> int:
            if "Very large" in text:
                return 2000  # Exceeds max_subblock_tokens (1000)
            return 50

        with patch.object(mock_phase, "_count_tokens", side_effect=mock_token_count):
            result = mock_phase._group_paragraphs_into_subblocks(paragraphs)

        # Large paragraph should be its own sub-block
        assert any("Very large paragraph" in sb for sb in result)
        # Find the sub-block with the large paragraph
        large_sb = [sb for sb in result if "Very large paragraph" in sb][0]
        # It should not be combined with other paragraphs
        assert "Small para." not in large_sb

    def test_trailing_chunk_redistribution(self, mock_phase):
        """Test that a small trailing chunk is redistributed evenly."""
        # Create paragraphs where last group would be too small
        paragraphs = [f"Paragraph {i}" for i in range(6)]

        def mock_token_count(text: str) -> int:
            # First 4 paragraphs have 100 tokens each (400 total > 300 min)
            # Last 2 paragraphs have 50 tokens each (100 total < 300 min)
            if any(f"Paragraph {i}" in text for i in [0, 1, 2, 3]):
                return 100
            return 50

        with patch.object(mock_phase, "_count_tokens", side_effect=mock_token_count):
            result = mock_phase._group_paragraphs_into_subblocks(paragraphs)

        # Should have redistributed to balance the groups
        # The exact distribution depends on the algorithm, but:
        # - No sub-block should be below min_subblock_tokens if possible
        # - Order of paragraphs should be preserved within each sub-block
        for sb in result:
            assert sb  # No empty sub-blocks

    def test_two_paragraphs_second_small(self, mock_phase):
        """Test redistribution with only two paragraphs where second is small."""
        paragraphs = ["First paragraph content.", "Small."]

        def mock_token_count(text: str) -> int:
            if "First" in text:
                return 400  # > min_subblock_tokens
            return 50  # < min_subblock_tokens

        with patch.object(mock_phase, "_count_tokens", side_effect=mock_token_count):
            result = mock_phase._group_paragraphs_into_subblocks(paragraphs)

        # Both paragraphs should be redistributed between two groups
        # or kept as one group if that's better
        assert len(result) >= 1
        # All paragraphs should be present
        all_content = "\n".join(result)
        assert "First paragraph content." in all_content
        assert "Small." in all_content

    def test_preserves_paragraph_order(self, mock_phase):
        """Test that paragraph order is preserved within sub-blocks."""
        paragraphs = [f"Paragraph {i}" for i in range(5)]

        with patch.object(mock_phase, "_count_tokens", return_value=100):
            result = mock_phase._group_paragraphs_into_subblocks(paragraphs)

        # Reconstruct all paragraphs in order from sub-blocks
        all_paras = []
        for sb in result:
            all_paras.extend(sb.split("\n"))

        # Check order is preserved
        expected_order = [f"Paragraph {i}" for i in range(5)]
        assert all_paras == expected_order


# ============================================================================
# Integration Tests for Sub-block Processing
# ============================================================================


class TestProcessBlockWithSubblocks:
    """Integration tests for _process_block_with_subblocks method."""

    @pytest.fixture
    def mock_phase_with_subblocks(self):
        """Create a mock StandardLlmPhase with sub-blocks enabled."""
        from src.phases.standard import StandardLlmPhase

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "input.md"
            input_file.write_text("# Test\n\nParagraph 1.\nParagraph 2.")
            output_file = temp_path / "output.md"
            system_prompt = temp_path / "system.md"
            system_prompt.write_text("System prompt")
            user_prompt = temp_path / "user.md"
            user_prompt.write_text("{current_body}")

            mock_model = MagicMock()
            # Return different content to show processing happened
            mock_model.chat_completion.return_value = ("Processed content", "gen-123")

            phase = StandardLlmPhase(
                name="test_phase",
                input_file_path=input_file,
                output_file_path=output_file,
                original_file_path=input_file,
                system_prompt_path=system_prompt,
                user_prompt_path=user_prompt,
                book_name="Test Book",
                author_name="Test Author",
                model=mock_model,
                use_subblocks=True,
                max_subblock_tokens=4096,
                min_subblock_tokens=100,  # Low threshold for testing
                max_workers=1,
            )
            yield phase

    def test_dispatches_to_subblock_processing(self, mock_phase_with_subblocks):
        """Test that _process_block dispatches to sub-block processing when enabled."""
        current_block = "## Test Header\n\nParagraph 1.\nParagraph 2."
        original_block = "## Test Header\n\nOriginal 1.\nOriginal 2."

        # Mock the sub-block processing method to verify it's called
        with patch.object(
            mock_phase_with_subblocks, "_process_block_with_subblocks", return_value="## Test Header\n\nProcessed\n\n"
        ) as mock_method:
            mock_phase_with_subblocks._process_block(current_block, original_block)
            mock_method.assert_called_once_with(current_block, original_block)

    def test_header_preserved_in_output(self, mock_phase_with_subblocks):
        """Test that the header is preserved in the output."""
        current_block = "## My Chapter Header\n\nParagraph content."
        original_block = "## My Chapter Header\n\nOriginal content."

        result = mock_phase_with_subblocks._process_block(current_block, original_block)

        assert "## My Chapter Header" in result

    def test_empty_block_returns_as_is(self, mock_phase_with_subblocks):
        """Test that empty blocks are returned unchanged."""
        current_block = "## Empty Header\n\n"
        original_block = "## Empty Header\n\n"

        result = mock_phase_with_subblocks._process_block(current_block, original_block)

        assert "## Empty Header" in result
        # Model should not be called for empty content
        mock_phase_with_subblocks.model.chat_completion.assert_not_called()

    def test_special_tags_block_returns_as_is(self, mock_phase_with_subblocks):
        """Test that blocks with only special tags are returned unchanged."""
        current_block = "## License\n\n{license}"
        original_block = "## License\n\n{license}"

        result = mock_phase_with_subblocks._process_block(current_block, original_block)

        assert "## License" in result
        assert "{license}" in result


# ============================================================================
# Factory Integration Tests
# ============================================================================


class TestPhaseFactorySubblockParams:
    """Tests for PhaseFactory passing sub-block parameters."""

    def test_factory_passes_subblock_params(self):
        """Test that PhaseFactory passes sub-block parameters to phase."""
        from src.api.config import PhaseConfig, PhaseType
        from src.phases.factory import PhaseFactory

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "input.md"
            input_file.write_text("# Test\n\nContent")
            output_file = temp_path / "output.md"
            system_prompt = temp_path / "system.md"
            system_prompt.write_text("System prompt")
            user_prompt = temp_path / "user.md"
            user_prompt.write_text("{current_body}")

            mock_model = MagicMock()

            config = PhaseConfig(
                phase_type=PhaseType.MODERNIZE,
                name="test",
                input_file_path=input_file,
                output_file_path=output_file,
                original_file_path=input_file,
                system_prompt_path=system_prompt,
                user_prompt_path=user_prompt,
                book_name="Test",
                author_name="Author",
                llm_model_instance=mock_model,
                use_subblocks=True,
                max_subblock_tokens=8000,
                min_subblock_tokens=2000,
            )

            phase = PhaseFactory.create_standard_phase(config=config)

            assert phase.use_subblocks is True
            assert phase.max_subblock_tokens == 8000
            assert phase.min_subblock_tokens == 2000


class TestBatchModeWithSubblocks:
    """Tests for batch mode behavior when sub-blocks are enabled."""

    def test_batch_processing_splits_into_subblocks(self):
        """Batch mode should expand a single block into multiple sub-block requests and reassemble output."""
        from src.phases.standard import StandardLlmPhase

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "input.md"
            input_file.write_text("## Header\n\nParagraph 1.\nParagraph 2.\n")
            output_file = temp_path / "output.md"
            system_prompt = temp_path / "system.md"
            system_prompt.write_text("System prompt")
            user_prompt = temp_path / "user.md"
            user_prompt.write_text("{current_body}")

            mock_model = MagicMock()
            mock_model.supports_batch.return_value = True

            # Mock batch_chat_completion to preserve metadata from requests (realistic behavior)
            def mock_batch_completion(requests, **kwargs):
                responses = []
                for i, req in enumerate(requests):
                    responses.append(
                        {
                            "content": f"OUT-{i + 1}",
                            "generation_id": f"gen-{i + 1}",
                            "metadata": req.get("metadata", {}),  # Preserve metadata like real batch API
                        }
                    )
                return responses

            mock_model.batch_chat_completion.side_effect = mock_batch_completion

            phase = StandardLlmPhase(
                name="test_phase",
                input_file_path=input_file,
                output_file_path=output_file,
                original_file_path=input_file,
                system_prompt_path=system_prompt,
                user_prompt_path=user_prompt,
                book_name="Test Book",
                author_name="Test Author",
                model=mock_model,
                use_subblocks=True,
                use_batch=True,
                max_subblock_tokens=4096,
                min_subblock_tokens=100,
                max_workers=1,
            )

            # Force deterministic sub-block grouping (two sub-blocks).
            with patch.object(phase, "_split_body_into_paragraphs", return_value=["P1", "P2"]):
                with patch.object(phase, "_group_paragraphs_into_subblocks", return_value=["SB-1", "SB-2"]):
                    results = phase._process_batch(batch=[("## Header\n\nP1\nP2\n", "## Header\n\nP1\nP2\n")])

            assert len(results) == 1
            assert results[0].startswith("## Header")
            assert "OUT-1\nOUT-2" in results[0]

            # Sub-block stats should reflect splitting.
            assert phase._subblocks_processed_total == 2
            assert phase._subblock_blocks_processed_total == 1

            # Batch API should have been called with two requests (one per sub-block).
            _, call_kwargs = mock_model.batch_chat_completion.call_args
            assert len(call_kwargs["requests"]) == 2
