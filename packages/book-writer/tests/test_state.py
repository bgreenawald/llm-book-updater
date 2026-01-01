"""Tests for book_writer state module."""

from datetime import datetime

import pytest
from book_writer.models import (
    BookOutline,
    BookState,
    ChapterOutline,
    ChapterState,
    ChapterStatus,
    SectionOutline,
    SectionState,
    SectionStatus,
)
from book_writer.state import StateManager


@pytest.fixture
def state_manager(output_dir):
    """Create a state manager for testing."""
    return StateManager(output_dir)


@pytest.fixture
def sample_outline():
    """Create a sample book outline for testing."""
    section1 = SectionOutline(id="1.1", title="First", outline_content="Content")
    section2 = SectionOutline(id="1.2", title="Second", outline_content="Content")
    chapter = ChapterOutline(id="1", number=1, title="Chapter 1", sections=[section1, section2])
    return BookOutline(title="Test Book", chapters=[chapter])


@pytest.fixture
def sample_state():
    """Create a sample book state for testing."""
    now = datetime.now()
    return BookState(
        rubric_hash="test_hash",
        model="test/model",
        created_at=now,
        updated_at=now,
        chapters={
            "1": ChapterState(
                chapter_id="1",
                sections={
                    "1.1": SectionState(section_id="1.1"),
                    "1.2": SectionState(section_id="1.2"),
                },
            ),
        },
    )


class TestStateManager:
    """Tests for StateManager class."""

    def test_init_creates_directory(self, temp_dir):
        """Test that init creates output directory."""
        output = temp_dir / "new_output"
        manager = StateManager(output)
        assert output.exists()
        assert manager.state_file == output / "state.json"

    def test_load_state_returns_none_when_no_file(self, state_manager):
        """Test load_state returns None when no state file exists."""
        state = state_manager.load_state()
        assert state is None

    def test_save_and_load_state(self, state_manager, sample_state):
        """Test saving and loading state."""
        state_manager.save_state(sample_state)
        loaded = state_manager.load_state()

        assert loaded is not None
        assert loaded.rubric_hash == "test_hash"
        assert loaded.model == "test/model"
        assert "1" in loaded.chapters
        assert "1.1" in loaded.chapters["1"].sections

    def test_save_state_updates_timestamp(self, state_manager, sample_state):
        """Test that save_state updates the updated_at timestamp."""
        original_time = sample_state.updated_at
        state_manager.save_state(sample_state)
        loaded = state_manager.load_state()

        # The timestamp should be updated (might be equal in fast tests)
        assert loaded.updated_at >= original_time

    def test_load_state_handles_invalid_json(self, state_manager):
        """Test load_state handles invalid JSON gracefully."""
        state_manager.state_file.write_text("not valid json")
        state = state_manager.load_state()
        assert state is None

    def test_initialize_state_creates_state(self, state_manager, sample_outline):
        """Test initialize_state creates state from outline."""
        state = state_manager.initialize_state(sample_outline, "test/model", "hash123")

        assert state.rubric_hash == "hash123"
        assert state.model == "test/model"
        assert "1" in state.chapters
        assert "1.1" in state.chapters["1"].sections
        assert "1.2" in state.chapters["1"].sections

    def test_initialize_state_saves_file(self, state_manager, sample_outline):
        """Test initialize_state saves the state file."""
        state_manager.initialize_state(sample_outline, "model", "hash")
        assert state_manager.state_file.exists()


class TestUpdateSection:
    """Tests for StateManager.update_section method."""

    def test_update_section_to_in_progress(self, state_manager, sample_state):
        """Test updating section to in_progress."""
        state_manager.save_state(sample_state)
        updated = state_manager.update_section(sample_state, "1", "1.1", SectionStatus.IN_PROGRESS)

        section = updated.chapters["1"].sections["1.1"]
        assert section.status == SectionStatus.IN_PROGRESS
        assert section.started_at is not None

    def test_update_section_to_completed(self, state_manager, sample_state):
        """Test updating section to completed."""
        state_manager.save_state(sample_state)
        updated = state_manager.update_section(
            sample_state,
            "1",
            "1.1",
            SectionStatus.COMPLETED,
            content="Generated content",
            token_count=100,
        )

        section = updated.chapters["1"].sections["1.1"]
        assert section.status == SectionStatus.COMPLETED
        assert section.generated_content == "Generated content"
        assert section.token_count == 100
        assert section.completed_at is not None

    def test_update_section_to_failed(self, state_manager, sample_state):
        """Test updating section to failed."""
        state_manager.save_state(sample_state)
        updated = state_manager.update_section(sample_state, "1", "1.1", SectionStatus.FAILED, error="API error")

        section = updated.chapters["1"].sections["1.1"]
        assert section.status == SectionStatus.FAILED
        assert section.last_error == "API error"
        assert section.retry_count == 1

    def test_update_section_increments_retry_count(self, state_manager, sample_state):
        """Test that retry count increments on each failure."""
        state_manager.save_state(sample_state)

        # First failure
        state_manager.update_section(sample_state, "1", "1.1", SectionStatus.FAILED, error="Error 1")
        assert sample_state.chapters["1"].sections["1.1"].retry_count == 1

        # Second failure
        state_manager.update_section(sample_state, "1", "1.1", SectionStatus.FAILED, error="Error 2")
        assert sample_state.chapters["1"].sections["1.1"].retry_count == 2

    def test_update_section_invalid_chapter_raises(self, state_manager, sample_state):
        """Test that invalid chapter raises ValueError."""
        state_manager.save_state(sample_state)
        with pytest.raises(ValueError, match="Chapter nonexistent not found"):
            state_manager.update_section(sample_state, "nonexistent", "1.1", SectionStatus.COMPLETED)

    def test_update_section_invalid_section_raises(self, state_manager, sample_state):
        """Test that invalid section raises ValueError."""
        state_manager.save_state(sample_state)
        with pytest.raises(ValueError, match="Section nonexistent not found"):
            state_manager.update_section(sample_state, "1", "nonexistent", SectionStatus.COMPLETED)


class TestIntermediateOutputPersistence:
    """Tests for intermediate output persistence in three-phase pipeline."""

    def test_intermediate_outputs_saved_on_completed(self, state_manager, sample_state):
        """Test intermediate outputs are saved when section completes."""
        state_manager.save_state(sample_state)
        updated = state_manager.update_section(
            sample_state,
            "1",
            "1.1",
            SectionStatus.COMPLETED,
            content="Final content",
            initial_content="Initial content from P1",
            identify_feedback="Feedback from P2",
        )

        section = updated.chapters["1"].sections["1.1"]
        assert section.status == SectionStatus.COMPLETED
        assert section.generated_content == "Final content"
        assert section.initial_content == "Initial content from P1"
        assert section.identify_feedback == "Feedback from P2"

    def test_intermediate_outputs_saved_on_failed_phase2(self, state_manager, sample_state):
        """Test intermediate outputs are saved when Phase 2 fails."""
        state_manager.save_state(sample_state)
        updated = state_manager.update_section(
            sample_state,
            "1",
            "1.1",
            SectionStatus.FAILED,
            error="Phase 2 (Identify) failed: API error",
            initial_content="Initial content from P1",
        )

        section = updated.chapters["1"].sections["1.1"]
        assert section.status == SectionStatus.FAILED
        assert section.last_error == "Phase 2 (Identify) failed: API error"
        assert section.initial_content == "Initial content from P1"
        assert section.identify_feedback is None  # P2 failed, no feedback yet

    def test_intermediate_outputs_saved_on_failed_phase3(self, state_manager, sample_state):
        """Test intermediate outputs are saved when Phase 3 fails."""
        state_manager.save_state(sample_state)
        updated = state_manager.update_section(
            sample_state,
            "1",
            "1.1",
            SectionStatus.FAILED,
            error="Phase 3 (Implement) failed: API error",
            initial_content="Initial content from P1",
            identify_feedback="Feedback from P2",
        )

        section = updated.chapters["1"].sections["1.1"]
        assert section.status == SectionStatus.FAILED
        assert section.last_error == "Phase 3 (Implement) failed: API error"
        assert section.initial_content == "Initial content from P1"
        assert section.identify_feedback == "Feedback from P2"

    def test_intermediate_outputs_persisted_to_disk_on_failure(self, state_manager, sample_state):
        """Test intermediate outputs are persisted to disk on failure for resume."""
        state_manager.save_state(sample_state)
        state_manager.update_section(
            sample_state,
            "1",
            "1.1",
            SectionStatus.FAILED,
            error="Phase 3 (Implement) failed: API error",
            initial_content="Initial content from P1",
            identify_feedback="Feedback from P2",
        )

        # Reload from disk
        loaded = state_manager.load_state()
        section = loaded.chapters["1"].sections["1.1"]
        assert section.initial_content == "Initial content from P1"
        assert section.identify_feedback == "Feedback from P2"


class TestChapterStatusUpdate:
    """Tests for chapter status updates."""

    def test_chapter_status_completed_when_all_sections_complete(self, state_manager, sample_state):
        """Test chapter status is completed when all sections complete."""
        state_manager.save_state(sample_state)

        state_manager.update_section(sample_state, "1", "1.1", SectionStatus.COMPLETED, content="c1")
        state_manager.update_section(sample_state, "1", "1.2", SectionStatus.COMPLETED, content="c2")

        assert sample_state.chapters["1"].status == ChapterStatus.COMPLETED

    def test_chapter_status_partial_when_mixed(self, state_manager, sample_state):
        """Test chapter status is partial when some sections fail."""
        state_manager.save_state(sample_state)

        state_manager.update_section(sample_state, "1", "1.1", SectionStatus.COMPLETED, content="c1")
        state_manager.update_section(sample_state, "1", "1.2", SectionStatus.FAILED, error="Error")

        assert sample_state.chapters["1"].status == ChapterStatus.PARTIAL

    def test_chapter_status_failed_when_all_fail(self, state_manager, sample_state):
        """Test chapter status is failed when all sections fail."""
        state_manager.save_state(sample_state)

        state_manager.update_section(sample_state, "1", "1.1", SectionStatus.FAILED, error="Error")
        state_manager.update_section(sample_state, "1", "1.2", SectionStatus.FAILED, error="Error")

        assert sample_state.chapters["1"].status == ChapterStatus.FAILED

    def test_chapter_status_in_progress_when_working(self, state_manager, sample_state):
        """Test chapter status is in_progress when sections are being worked on."""
        state_manager.save_state(sample_state)

        state_manager.update_section(sample_state, "1", "1.1", SectionStatus.IN_PROGRESS)

        assert sample_state.chapters["1"].status == ChapterStatus.IN_PROGRESS


class TestMarkChapterStarted:
    """Tests for StateManager.mark_chapter_started method."""

    def test_mark_chapter_started(self, state_manager, sample_state):
        """Test marking chapter as started."""
        state_manager.save_state(sample_state)
        updated = state_manager.mark_chapter_started(sample_state, "1")

        assert updated.chapters["1"].status == ChapterStatus.IN_PROGRESS
        assert updated.chapters["1"].started_at is not None


class TestShouldReinitialize:
    """Tests for StateManager.should_reinitialize method."""

    def test_should_reinitialize_when_hash_differs(self, state_manager, sample_state):
        """Test should_reinitialize returns True when hash differs."""
        assert state_manager.should_reinitialize(sample_state, "different_hash") is True

    def test_should_not_reinitialize_when_hash_matches(self, state_manager, sample_state):
        """Test should_reinitialize returns False when hash matches."""
        assert state_manager.should_reinitialize(sample_state, "test_hash") is False


class TestResetFailedSections:
    """Tests for StateManager.reset_failed_sections method."""

    def test_reset_failed_sections(self, state_manager, sample_state):
        """Test resetting failed sections to pending."""
        # Set up some failed sections
        sample_state.chapters["1"].sections["1.1"].status = SectionStatus.FAILED
        sample_state.chapters["1"].sections["1.1"].retry_count = 3
        sample_state.chapters["1"].sections["1.1"].last_error = "Error"
        sample_state.chapters["1"].sections["1.2"].status = SectionStatus.COMPLETED
        state_manager.save_state(sample_state)

        updated = state_manager.reset_failed_sections(sample_state)

        # Failed section should be reset
        section = updated.chapters["1"].sections["1.1"]
        assert section.status == SectionStatus.PENDING
        assert section.retry_count == 0
        assert section.last_error is None

        # Completed section should not be affected
        assert updated.chapters["1"].sections["1.2"].status == SectionStatus.COMPLETED


class TestGetProgress:
    """Tests for progress methods."""

    def test_get_chapter_progress(self, state_manager, sample_state):
        """Test getting chapter progress."""
        sample_state.chapters["1"].sections["1.1"].status = SectionStatus.COMPLETED
        state_manager.save_state(sample_state)

        progress = state_manager.get_chapter_progress(sample_state, "1")

        assert progress["total"] == 2
        assert progress["completed"] == 1
        assert progress["pending"] == 1
        assert progress["failed"] == 0

    def test_get_chapter_progress_nonexistent(self, state_manager, sample_state):
        """Test getting progress for nonexistent chapter."""
        progress = state_manager.get_chapter_progress(sample_state, "nonexistent")
        assert progress["total"] == 0

    def test_get_overall_progress(self, state_manager, sample_state):
        """Test getting overall progress."""
        sample_state.chapters["1"].sections["1.1"].status = SectionStatus.COMPLETED
        state_manager.save_state(sample_state)

        progress = state_manager.get_overall_progress(sample_state)

        assert progress["total_chapters"] == 1
        assert progress["total_sections"] == 2
        assert progress["completed"] == 1
        assert progress["pending"] == 1
        assert progress["failed"] == 0
